import yaml
import os
import pandas as pd
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate csv file containing metadata of models')
    parser.add_argument('directory', help='Directory containing the metafiles')
    parser.add_argument('out', help='Output .csv file')
    parser.add_argument('--dataset', type=str, help='Models trained on specified dataset. Defaults to "ImageNet-1k"',
                        default='ImageNet-1k')
    parser.add_argument('--task', type=str,
                        help='Models trained for specified task. Defaults to "Image Classification"',
                        default='Image Classification')
    arguments = parser.parse_args()
    return arguments


def main(directory, out_file, dataset: str, task: str):

    file_paths = []
    # Iterate over subdirectories and files
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file is a metafile.yml
            if file == 'metafile.yml':
                # Get the full path of the metafile.yml
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

    prog_bar = tqdm(total=len(file_paths), desc="Looking through metafiles.....", unit='item')
    dataframe = []
    sort_keys = []

    for file_path in file_paths:
        # Read the contents of the metafile.yml
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        for model in data['Models']:
            try:
                if model['Results'] is not None:
                    try:
                        for result in model['Results']:
                            if result['Dataset'] == dataset and result['Task'] == task:
                                try:
                                    row = {
                                        'Name': model['Name'],
                                        'Config': model['Config']
                                    }
                                    for key, val in model['Metadata'].items():
                                        if key == 'Training Data':
                                            continue
                                        row[key] = val
                                    for key, val in result['Metrics'].items():
                                        row[key] = val
                                        sort_keys.append(key)
                                    dataframe.append(row)
                                except KeyError:
                                    continue
                    except TypeError:
                        result = model['Results']
                        if result['Dataset'] == dataset and result['Task'] == task:
                            try:
                                row = {
                                    'Name': model['Name'],
                                    'Config': model['Config']
                                }
                                for key, val in model['Metadata'].items():
                                    if key == 'Training Data':
                                        continue
                                    row[key] = val
                                for key, val in result['Metrics'].items():
                                    row[key] = val
                                    sort_keys.append(key)
                                dataframe.append(row)
                            except KeyError:
                                continue
            except KeyError:
                continue

        prog_bar.update(1)
    prog_bar.close()

    sort_keys = set(sort_keys)
    if 'Top 1 Accuracy' in sort_keys:
        sort_key = 'Top 1 Accuracy'
    else:
        sort_key = list(sort_keys)[0]

    df = pd.DataFrame(dataframe).sort_values(by=sort_key, ascending=False)
    df.to_csv(out_file, index=False)


if __name__ == '__main__':
    args = parse_args()
    if args.directory.split('/')[0] == 'mmaction2':
        tasks = ('Action Detection', 'Temporal Action Localization', 'Action Recognition', 'Skeleton-based Action Recognition')
        datasets = ('Kinetics-600', 'AVA v2.2', 'SthV2', 'HMDB51', 'NTU60-XSub', 'AVA v2.1', 'NTU120-XSub-3D',
                    'ActivityNet v1.3', 'Moments in Time V1', 'NTU120-XSub-2D', 'Kinetics-700', 'Kinetics-400',
                    'FineGYM', 'NTU60-XSub-2D', 'NTU60-XSub-3D', 'UCF101', 'SthV1')
        assert args.task in tasks, f"task must be one of {tasks}"
        assert args.dataset in datasets, f"dataset must be one of {datasets}"
    if args.directory.split('/')[0] == 'mmpretrain':
        tasks = ('Image-To-Text Retrieval', 'Visual Question Answering', 'Image Retrieval', 'Image Classification',
                 'Image Caption', 'Multi-Label Classification', 'NLVR', 'Text-To-Image Retrieval', 'Visual Grounding')
        datasets = ('CIFAR100', 'ImageNet-1k', 'CIFAR-10', 'CUB-200-2011', 'RefCOCO', 'NLVR2', 'VQAv2', 'CIFAR-100',
                    'COCO', 'InShop', 'PASCAL VOC 2007')
        assert args.task in tasks, f"task must be one of {tasks}"
        assert args.dataset in datasets, f"dataset must be one of {datasets}"

    main(args.directory, args.out, args.dataset, args.task)
