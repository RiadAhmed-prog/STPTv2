import os.path
import pickle
import time
from tqdm import tqdm
from tools.stpt_utils import *
from dotenv import load_dotenv
import argparse

load_dotenv()
# Define the ANSI escape code for yellow color
YELLOW = '\033[93m'

ROOT_DIR = os.getenv('ROOT_DIR')

groups = ['medical-conditions', 'daily-actions', 'mutual-actions']


def parse_args():
    parser = argparse.ArgumentParser(description='Generate STPT features and save them')
    parser.add_argument('--group', type=str, default=None,
                        choices=['medical-conditions', 'daily-actions', 'mutual-actions'])
    parser.add_argument('--cls-input', nargs='+', type=str, help='Select action classes')
    parser.add_argument('--limit', type=int, help='max examples to store')
    parser.add_argument('--features', nargs='+', type=str, help='features to generate')
    parser.add_argument('--dump-pickle', type=str, help='/path/to/pickle for dumping feature points', default=None)

    args = parser.parse_args()
    return args


def take_input_from_user():
    group = str(input('Select action group: ')) or 'ntu120'
    assert group in groups, f"{group} must be one of {groups}"
    cls_input = input(f"Select action class(s) from {group} (separated by comma): ")
    cls_input = [c for c in cls_input.split(', ')]
    limit = int(input('Select amount of examples per class: '))
    features = input('Name of features to calculate (separated by comma): ')
    features = [f for f in features.split(', ')]
    dump_pickle = input('Specify /path/to/pickle for dumping feature points: ') or None

    return argparse.Namespace(group=group, cls_input=cls_input,
                              limit=limit, features=features, dump_pickle=dump_pickle)


def main(args):
    if args.group == 'ntu120':
        pickle_file = os.path.join(ROOT_DIR, 'data/skeleton', f"ntu120_2d.pkl")
        group = 'ntu120'
        classes = load_label_map(os.path.join(ROOT_DIR, "misc", f"label_map_{group}.txt"))
    else:
        pickle_file = os.path.join(ROOT_DIR, 'data/skeleton', f"ntu120_2d_{args.group}.pkl")
        if not os.path.exists(pickle_file):
            print(FileNotFoundError(f"{pickle_file} does not exist. Using ntu120_2d.pkl..."))
            pickle_file = os.path.join(ROOT_DIR, 'data/skeleton', f"ntu120_2d.pkl")
        classes = load_label_map(os.path.join(ROOT_DIR, "misc", f"{args.group}.txt"))

    print(f"Loading {pickle_file} into memory.....")
    start_time = time.time()
    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(
            f"{pickle_file} does not exist. Create or download from https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu120_2d.pkl")
    print(f"Successfully loaded in {time.time() - start_time: 0.2f}s")

    missing_data = load_missing_data(os.path.join(ROOT_DIR, "misc", "NTU_RGBD120_samples_with_missing_skeletons.txt"))

    selected_classes = []
    for item in args.cls_input:
        try:
            item = int(item)
            try:
                selected_classes.append(list(classes.keys()).index(item))
            except KeyError:
                print(YELLOW + UserWarning(f"{item} does not belong to {args.group}").__str__() + '\033[0m')
                continue
        except ValueError:
            key = find_key_by_value(classes, item)
            if key is None:
                print(YELLOW + UserWarning(f"{item} does not belong to {args.group}").__str__() + '\033[0m')
                continue
            selected_classes.append(list(classes.keys()).index(key))

    if len(selected_classes) == 0:
        print(YELLOW + Warning(f"selecting {classes.values()}").__str__() + '\033[0m')
        selected_classes = [cls for cls in classes.keys()]

    example_data = {
        'annotations': []
    }

    count = {f: 0 for f in selected_classes}
    bar = tqdm(total=len(data['annotations']), desc='Tracking annotations...', unit='item')
    for i, annotation in enumerate(data['annotations']):
        if annotation['frame_dir'] in missing_data:
            continue
        label = annotation['label']
        if label in selected_classes and count[label] < args.limit:
            annotation['features'] = args.features
            for feature in args.features:
                f_array = calculate_features(annotation, feature=feature)
                annotation[feature] = f_array
            example_data['annotations'].append(annotation)
            count[label] += 1

        bar.update(1)
    bar.close()

    with open(os.path.join(ROOT_DIR, "misc", f"{args.group}_example.pkl"), 'wb') as f:
        pickle.dump(example_data, f)
        print(f"examples saved to {os.path.join(ROOT_DIR, 'misc', f'{args.group}_example.pkl')}")

    if args.dump_pickle is not None:
        with open(args.dump_pickle, 'wb') as f:
            pickle.dump(data, f)
            print(f"feature data saved to {args.dump_pickle}")


if __name__ == '__main__':
    args = parse_args()
    if args.group is None:
        args = take_input_from_user()
    main(args)
