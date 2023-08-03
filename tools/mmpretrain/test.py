# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from copy import deepcopy
from datetime import datetime
import mmengine
from mmpretrain.evaluation import ConfusionMatrix
from mmpretrain.registry import DATASETS
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.evaluator import DumpResults
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.analysis import get_model_complexity_info
from mmpretrain import get_model

import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import json
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.getenv('ROOT_DIR')


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMPreTrain test (and eval) a model')
    parser.add_argument('--json', type=str, nargs='?', help='parse arguments from a JSON file')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='the file to output results.')
    parser.add_argument(
        '--out-item',
        choices=['metrics', 'pred'],
        help='To output whether metrics or predictions. '
             'Defaults to output predictions.')
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
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision test')
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
        '--no-pin-memory',
        action='store_true',
        help='whether to disable the pin_memory option in dataloaders.')
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Whether to enable the Test-Time-Aug (TTA). If the config file '
             'has `tta_pipeline` and `tta_model` fields, use them to determine the '
             'TTA transforms and how to merge the TTA results. Otherwise, use flip '
             'TTA by averaging classification score.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1920, 1080],
        help='input image size')
    parser.add_argument('--log-to-gsheets', nargs='+', type=str,
                        help='log test results to Google sheets. Arguments should be "{file name/sheet_no(1,2,etc.)}" "{path/to/client_secret.json}" "{gdrive folder id}"')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.json:
        with open(os.path.join(ROOT_DIR, args.json), 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)

    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('../../work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # enable automatic-mixed-precision test
    if args.amp:
        cfg.test_cfg.fp16 = True

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

    # -------------------- TTA related args --------------------
    if args.tta:
        if 'tta_model' not in cfg:
            cfg.tta_model = dict(type='mmpretrain.AverageClsScoreTTA')
        if 'tta_pipeline' not in cfg:
            test_pipeline = cfg.test_dataloader.dataset.pipeline
            cfg.tta_pipeline = deepcopy(test_pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [test_pipeline[-1]],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # ----------------- Default dataloader args -----------------
    default_dataloader_cfg = ConfigDict(
        pin_memory=True,
        collate_fn=dict(type='default_collate'),
    )

    def set_default_dataloader_cfg(cfg, field):
        if cfg.get(field, None) is None:
            return
        dataloader_cfg = deepcopy(default_dataloader_cfg)
        dataloader_cfg.update(cfg[field])
        cfg[field] = dataloader_cfg
        if args.no_pin_memory:
            cfg[field]['pin_memory'] = False

    set_default_dataloader_cfg(cfg, 'test_dataloader')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def get_flops(config, shape=(1920, 1080), ):
    if len(shape) == 1:
        input_shape = (3, shape[0], shape[0])
    elif len(shape) == 2:
        input_shape = (3,) + tuple(shape)
    else:
        raise ValueError('invalid input shape')
    model = get_model(config)
    model.eval()
    if hasattr(model, 'extract_feat'):
        model.forward = model.extract_feat
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))
    analysis_results = get_model_complexity_info(
        model,
        input_shape,
    )
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
            model = cfg.model.backbone.type
            try:
                ext = cfg.model.backbone.arch
            except AttributeError:
                try:
                    ext = cfg.model.backbone.depth
                except AttributeError:
                    ext = None

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
            dataset = '/'.join(cfg.train_dataloader.dataset.data_root.split('/')[1:])
            last_row = len(sheet.col_values(4))
            sheet.update_cell(last_row + 1, 4, dataset)

        if row == 'FLOPs':
            last_row = len(sheet.col_values(10))
            sheet.update_cell(last_row + 1, 10, flops)

        if row == 'Parameters':
            last_row = len(sheet.col_values(11))
            sheet.update_cell(last_row + 1, 11, params)

    for name, val in metrics.items():
        if name == 'accuracy/top1':
            last_row = len(sheet.col_values(5))
            sheet.update_cell(last_row + 1, 5, f'{val:.02f}%')
        if name == 'accuracy/top5':
            last_row = len(sheet.col_values(6))
            sheet.update_cell(last_row + 1, 6, f'{val:.02f}%')
        if name == 'single-label/precision':
            last_row = len(sheet.col_values(7))
            sheet.update_cell(last_row + 1, 7, f'{val:.02f}%')
        if name == 'single-label/recall':
            last_row = len(sheet.col_values(8))
            sheet.update_cell(last_row + 1, 8, f'{val:.02f}%')
        if name == 'confusion_matrix/result':
            cm_path = os.path.join(cfg.work_dir, 'confusion_matrix.png')
            last_row = len(sheet.col_values(9))
            # Upload the image to Google Drive and get the image URL
            image_metadata = {'name': f'{model}.png', 'parents': [folder_id]}
            media = MediaFileUpload(cm_path, mimetype='image/png')
            image_file = drive_service.files().create(body=image_metadata, media_body=media).execute()
            file_id = image_file['id']
            image_url = f"https://drive.google.com/uc?export=view&id={file_id}"
            # Update the cell with the image formula
            sheet.update_cell(last_row + 1, 9, f'=HYPERLINK("{image_url}", "Click here to open")')


def main(args):

    if args.out is None and args.out_item is not None:
        raise ValueError('Please use `--out` argument to specify the '
                         'path of the output file before using `--out-item`.')

    # load config
    cfg = Config.fromfile(args.config)

    # merge cli arguments to config
    cfg = merge_args(cfg, args)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    if args.out and args.out_item in ['pred', None]:
        runner.test_evaluator.metrics.append(
            DumpResults(out_file_path=args.out))

    # start testing
    metrics = runner.test()

    if args.out and 'metrics' in args.out_item:
        if 'confusion_matrix/result' in metrics.keys():
            try:
                # Try to build the dataset.
                dataset = DATASETS.build({
                    **cfg.test_dataloader.dataset, 'pipeline': []
                })
                classes = dataset.metainfo.get('classes')
            except Exception:
                classes = None
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

        mmengine.dump(metrics, args.out)

    if args.log_to_gsheets:
        log_to_gsheets(cfg, metrics, args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
