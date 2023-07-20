import os.path
from stpt_utils import *
import time
import pickle
from tqdm import tqdm

mising_data = []
with open('misc/NTU_RGBD120_samples_with_missing_skeletons.txt', "r") as file:
    for line in file:
        line = line.strip()
        mising_data.append(line)

pickle_file = 'ntu120_2d.pkl'
print(f"Loading {pickle_file} into memory...........")
start = time.time()
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)
print(f'Took {time.time() - start: 0.2f}s')

SAVE_DIR = 'data/STPT/examples_v2'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

i = 0
bar = tqdm(total=120, desc="Generating examples...", unit='item')
for annotation in data['annotations']:
    if annotation['frame_dir'] in mising_data:
        continue
    if annotation['label'] == i:
        output_path = os.path.join(SAVE_DIR, f'{annotation["frame_dir"]}.png')
        output_video = os.path.join(SAVE_DIR, f'{annotation["frame_dir"]}.mp4')
        generate_frame_plot(annotation, output_path)
        generate_kp_video(annotation, output_video)
        bar.update(1)
        i += 1
    if i == 120:
        break
bar.close()