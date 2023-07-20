import pickle
import os
import time
from stpt_utils import *
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate STPT dataset for NTU-RGB120')
    parser.add_argument('--group', type=str, help='Group into subset', default=None)
    parser.add_argument('--test', action='store_true', help='Create test set', default=False)
    arguments = parser.parse_args()
    return arguments


def main(group=None, test=False):
    groups = ['medical_conditions', 'daily_actions', 'mutual_actions']
    if group is not None:
        assert group in groups, \
            f"group must be one of {groups}"
        print(f'Creating subset for {group}')
        classes = load_label_map(f'misc/{group}.txt')
        mapped_classes = [cls for cls in classes.keys()]
    else:
        classes = [i for i in range(120)]

    mising_data = []
    with open('misc/NTU_RGBD120_samples_with_missing_skeletons.txt', "r") as file:
        for line in file:
            line = line.strip()
            mising_data.append(line)

    if not test:
        splits = {'train': [], 'val': []}
    else:
        splits = {'train': [], 'val': [], 'test': []}
    if group is None:
        DATA_DIR = os.path.join('data/STPT', 'NTU120')
    else:
        DATA_DIR = os.path.join('data/STPT', group)

    meta_dir = os.path.join(DATA_DIR, 'meta')
    if not os.path.exists(meta_dir):
        os.makedirs(meta_dir)

    pickle_file = 'ntu120_2d.pkl'
    print(f"Loading {pickle_file} into memory...........")
    start = time.time()
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    print(f'Took {time.time() - start: 0.2f}s')

    track_bar = tqdm(total=len(data['annotations']), desc=f"Tracking annotations.....", unit='item')
    for annotation in data['annotations']:
        if annotation['frame_dir'] in mising_data:
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
                label = mapped_classes.index(label)
            generate_frame_plot(annotation, output_file)
            f.write(f"{annotation['frame_dir']}.png {label}\n")
            progress_bar.update(1)
        progress_bar.close()
        del progress_bar


if __name__ == '__main__':
    args = parse_args()
    main(args.group, args.test)
