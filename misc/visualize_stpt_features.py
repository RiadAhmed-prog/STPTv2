import os.path
import pickle
import cv2
import argparse
from tools.stpt_utils import generate_video
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.getenv('ROOT_DIR')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate STPT dataset for NTU-RGB120')
    parser.add_argument('dataset', type=str, help='pickle file containing examples')
    parser.add_argument('--out-folder', type=str, help='folder to save output videos', default=None)
    parser.add_argument('--show', action='store_true', help='show frame by frame', default=False)
    arguments = parser.parse_args()
    return arguments


def main(path, out_folder=None, show=False):
    if out_folder is not None:
        out_folder = os.path.join(ROOT_DIR, out_folder)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

    with open(os.path.join(ROOT_DIR, path), 'rb') as f:
        data = pickle.load(f)

    for annotation in data['annotations']:
        if out_folder is not None:
            out_filename = os.path.join(ROOT_DIR, out_folder, f"{annotation['frame_dir']}.mp4")
        else:
            out_filename = None
        generate_video(annotation, out_filename, show, features=annotation['features'])


if __name__ == '__main__':
    args = parse_args()
    main(args.dataset, args.out_folder, args.show)
