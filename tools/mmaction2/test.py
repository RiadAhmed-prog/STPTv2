# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmaction.evaluation import ConfusionMatrix
from mmengine.registry import init_default_scope

from mmaction.registry import MODELS

try:
    from mmengine.analysis import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')

from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--dump',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=2,
        help='display time of every window. (second)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1, 17, 100, 64, 64],
        help='input image size')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--log-to-gsheets', nargs='+', type=str,
                        help='log test results to Google sheets. Arguments should be "{sheets name/int(sheet)}" "{path/to/client_secret.json}" "{gdrive folder id}"')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    # -------------------- visualization --------------------
    if args.show or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks, \
            'VisualizationHook is not set in the `default_hooks` field of ' \
            'config. Please set `visualization=dict(type="VisualizationHook")`'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = args.show
        cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    # -------------------- Dump predictions --------------------
    if args.dump is not None:
        assert args.dump.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        dump_metric = dict(type='DumpResults', out_file_path=args.dump)
        if isinstance(cfg.test_evaluator, (list, tuple)):
            cfg.test_evaluator = list(cfg.test_evaluator)
            cfg.test_evaluator.append(dump_metric)
        else:
            cfg.test_evaluator = [cfg.test_evaluator, dump_metric]

    return cfg


def get_flops(cfg, shape=(1, 17, 100, 64, 64)):
    if len(shape) == 1:
        input_shape = (1, 3, shape[0], shape[0])
    elif len(shape) == 2:
        input_shape = (1, 3) + tuple(shape)
    elif len(shape) == 4:
        # n, c, h, w = args.shape for 2D recognizer
        input_shape = tuple(shape)
    elif len(shape) == 5:
        # n, c, t, h, w = args.shape for 3D recognizer or
        # n, m, t, v, c = args.shape for GCN-based recognizer
        input_shape = tuple(shape)
    else:
        raise ValueError('invalid input shape')

    init_default_scope(cfg.get('default_scope', 'mmaction'))
    model = MODELS.build(cfg.model)
    model.eval()

    if hasattr(model, 'extract_feat'):
        model.forward = model.extract_feat
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    analysis_results = get_model_complexity_info(model, input_shape)
    flops = analysis_results['flops_str']
    params = analysis_results['params_str']

    return flops, params


def log_to_gsheets(cfg, metrics, args):
    [sheets_name, client_secret_json, folder_id] = args.log_to_gsheets
    assert sheets_name != '' and client_secret_json != '' and folder_id != '', \
        "Need sheets file name, path to client_secret.json and gdrive folder id"

    print(f'Writing results to Google Sheets file......')
    # define scope
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']

    # set credentials
    creds = Credentials.from_service_account_file(client_secret_json, scopes=scope)
    drive_service = build('drive', 'v3', credentials=creds)

    # authorize access to Google Sheets API
    client = gspread.authorize(creds)

    # open the Google Sheet
    file = sheets_name.split('/')[0]
    sheet_no = int(sheets_name.split('/')[1]) - 1
    sheet = client.open(file).get_worksheet(sheet_no)
    rows = sheet.row_values(1)

    if 'FLOPs' in rows or 'Parameters' in rows:
        assert args.shape, 'Need input shape for FLOPs and params'
        flops, params = get_flops(cfg, shape=args.shape)

    for row in rows:
        if row == 'Date':
            now = datetime.now()
            last_row = len(sheet.col_values(1))
            sheet.update_cell(last_row + 1, 1, f'{now.strftime("%d/%m/%Y %H:%M:%S")}')

        if row == 'Model':
            model = cfg.model.type
            try:
                ext = f"{cfg.model.backbone.type}-{cfg.model.backbone.depth}"
            except AttributeError:
                try:
                    ext = f"{cfg.model.backbone.type}-{cfg.model.backbone.tcn_type}"
                except AttributeError:
                    ext = f"{cfg.model.backbone.type}"

            if ext is not None:
                model = f'{model}_{ext}'

            last_row = len(sheet.col_values(2))
            sheet.update_cell(last_row + 1, 2, f'{model}')

        if row == 'Config':
            # Upload the config file to Google Drive and get the URL
            file_metadata = {'name': os.path.basename(cfg.filename), 'parents': [folder_id]}
            file = MediaFileUpload(cfg.filename, mimetype=None)
            upload_file = drive_service.files().create(body=file_metadata, media_body=file).execute()
            file_id = upload_file['id']
            file_url = f"https://drive.google.com/uc?export=view&id={file_id}"
            # Update the cell with the hyperlink formula
            last_row = len(sheet.col_values(3))
            sheet.update_cell(last_row + 1, 3, f'=HYPERLINK("{file_url}", "{os.path.basename(cfg.filename)}")')

        if row == 'Dataset':
            dataset = cfg.val_dataloader.dataset.ann_file.split('/')[-1]
            last_row = len(sheet.col_values(4))
            sheet.update_cell(last_row + 1, 4, dataset)

        if row == 'FLOPs':
            last_row = len(sheet.col_values(9))
            sheet.update_cell(last_row + 1, 9, flops)

        if row == 'Parameters':
            last_row = len(sheet.col_values(10))
            sheet.update_cell(last_row + 1, 10, params)

    for name, val in metrics.items():
        if name == 'acc/top1':
            last_row = len(sheet.col_values(5))
            sheet.update_cell(last_row + 1, 5, f'{val * 100:.02f}%')
        if name == 'acc/top5':
            last_row = len(sheet.col_values(6))
            sheet.update_cell(last_row + 1, 6, f'{val * 100:.02f}%')
        if name == 'acc/mean1':
            last_row = len(sheet.col_values(7))
            sheet.update_cell(last_row + 1, 7, f'{val * 100:.02f}%')
        if name == 'confusion_matrix/result':
            cm_path = os.path.join(cfg.work_dir, 'confusion_matrix.png')
            last_row = len(sheet.col_values(8))
            # Upload the image to Google Drive and get the image URL
            image_metadata = {'name': f'{model}.png', 'parents': [folder_id]}
            media = MediaFileUpload(cm_path, mimetype='image/png')
            image_file = drive_service.files().create(body=image_metadata, media_body=media).execute()
            file_id = image_file['id']
            image_url = f"https://drive.google.com/uc?export=view&id={file_id}"
            # Update the cell with the image formula
            sheet.update_cell(last_row + 1, 8, f'=HYPERLINK("{image_url}", "Click here to open")')


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start testing
    metrics = runner.test()

    if 'confusion_matrix/result' in metrics.keys():
        classes = runner.test_loop.dataloader.dataset.metainfo.get(
            'classes', None)
        fig = ConfusionMatrix.plot(
            metrics['confusion_matrix/result'],
            show=False,
            classes=classes,
            include_values=True,
            cmap='viridis')
        cm_path = os.path.join(cfg.work_dir, 'confusion_matrix.png')
        fig.savefig(cm_path)
        print(f'The confusion matrix is saved at {cm_path}.')

        metrics['confusion_matrix/result'] = metrics['confusion_matrix/result'].detach().cpu().numpy()

    if args.log_to_gsheets:
        log_to_gsheets(cfg, metrics, args)


if __name__ == '__main__':
    main()
