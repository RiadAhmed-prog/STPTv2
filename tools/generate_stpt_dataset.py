import pickle
import os
import time
from stpt_utils import *
from tqdm import tqdm
import argparse
from mmengine.runner.checkpoint import (_load_checkpoint,
                                        _load_checkpoint_to_model)
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmaction.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate STPT dataset for NTU-RGB120')
    parser.add_argument('dataset', type=str, help='Choose between NTU120 or NTU60', choices=['NTU120', 'NTU60'])
    parser.add_argument('--group', type=str, help='Group into subset', default=None,
                        choices=['medical_conditions', 'daily_actions', 'mutual_actions'])
    parser.add_argument('--video', action='store_true', help='Generate keypoint video', default=False)
    parser.add_argument('--test', action='store_true', help='Generate testval set', default=False)
    parser.add_argument('--plot-skeleton-kpts', action='store_true', help='STPT using plot_skeleton_kpts', default=False)
    parser.add_argument('--stgcn-config', type=str, help='STGCN model config', default=None)
    parser.add_argument('--stgcn-weights', type=str, help='STGCN model weights', default=None)
    arguments = parser.parse_args()
    return arguments


def load_model(cfg, weights, device):
    init_default_scope(cfg.default_scope)

    model = MODELS.build(cfg.model)
    model.cfg = cfg

    # Load Checkpoint
    checkpoint = _load_checkpoint(weights, map_location='cpu')
    _load_checkpoint_to_model(model, checkpoint)
    model.to(device)
    model.eval()

    return model


# TODO: args in main, dataset with stpt features

def main(dataset, group=None, test=False, config=None, weights=None, video=False, plot_skeleton_kpts=False):
    if group is not None:
        print(f'Creating subset for {group}')
        classes = load_label_map(f'misc/{group}.txt')
        classes = [cls for cls in classes.keys()]
    else:
        classes = [i for i in range(120)]

    missing_data = []
    if dataset == 'NTU120':
        pickle_file = '../data/skeleton/ntu120_2d.pkl'
        file = '../misc/NTU_RGBD120_samples_with_missing_skeletons.txt'
    else:
        pickle_file = '../data/skeleton/ntu60_2d.pkl'
        file = '../misc/NTU_RGBD_samples_with_missing_skeletons.txt'
    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            missing_data.append(line)

    if config is not None and not plot_skeleton_kpts:
        root = 'data/STPTv2'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg = Config.fromfile(config)
        model = load_model(cfg, weights, device)
    elif config is None and plot_skeleton_kpts:
        root = 'data/STPT-skeleton'
    else:
        root = 'data/STPT'

    if not test:
        splits = {'train': [], 'val': []}
    else:
        splits = {'train': [], 'val': [], 'test': []}
    if group is None:
        DATA_DIR = os.path.join(root, dataset)
    else:
        assert dataset == 'NTU120', 'If group is set, dataset must be NTU120'
        if not test:
            DATA_DIR = os.path.join(root, group)
        else:
            DATA_DIR = os.path.join(root, f"{group}-testval")

    meta_dir = os.path.join(DATA_DIR, 'meta')
    if not os.path.exists(meta_dir):
        os.makedirs(meta_dir)
    if video:
        vid_dir = os.path.join(DATA_DIR, 'videos')
        if not os.path.exists(vid_dir):
            os.makedirs(vid_dir)

    print(f"Loading {pickle_file} into memory...........")
    start = time.time()
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    print(f'Took {time.time() - start: 0.2f}s')

    track_bar = tqdm(total=len(data['annotations']), desc=f"Tracking annotations.....", unit='item')
    for annotation in data['annotations']:
        if annotation['frame_dir'] in missing_data:
            track_bar.update(1)
            continue
        if group is not None:
            if int(annotation['label']) not in classes:
                track_bar.update(1)
                continue
        if annotation['frame_dir'] in data['split']['xsub_train']:
            splits['train'].append(annotation)
        if annotation['frame_dir'] in data['split']['xsub_val']:
            splits['val'].append(annotation)
        track_bar.update(1)
    track_bar.close()
    if test:
        num_len = len(splits['val']) // 2
        splits['test'] = splits['val'][-num_len:]
        splits['val'] = splits['val'][:-num_len]

    for split in splits.keys():
        if not os.path.exists(os.path.join(DATA_DIR, split)):
            os.makedirs(os.path.join(DATA_DIR, split))
        progress_bar = tqdm(total=len(splits[split]), desc=f"Generating {split} set.....", unit='item')
        f = open(os.path.join(meta_dir, f'{split}.txt'), 'a')
        for annotation in splits[split]:
            output_file = os.path.join(DATA_DIR, split, f"{annotation['frame_dir']}.png")
            label = annotation["label"]
            if group is not None:
                label = classes.index(label)
            if config is None and not plot_skeleton_kpts:
                generate_frame_plot(annotation, output_file)
            elif config is None and plot_skeleton_kpts:
                generate_frame_plot_kpts(annotation, output_file)
            else:
                generate_frame_plot_stgcn(cfg, model, device, annotation, output_file)
            f.write(f"{annotation['frame_dir']}.png {label}\n")
            if video:
                vid_file = os.path.join(vid_dir, split, f"{annotation['frame_dir']}.mp4")
                generate_video(annotation, vid_file)
            progress_bar.update(1)
        progress_bar.close()


if __name__ == '__main__':
    args = parse_args()
    main(args.dataset, args.group, args.test, args.stgcn_config, args.stgcn_weights, args.video, args.plot_skeleton_kpts)
