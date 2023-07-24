import os.path
from stpt_utils import *
import time
import pickle
from tqdm import tqdm
from mmengine.runner.checkpoint import (_load_checkpoint,
                                        _load_checkpoint_to_model)
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.dataset import Compose
import torch
from mmaction.registry import MODELS

flag = True

if flag:
    config = 'mmaction2/configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py'
    weights = 'https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221228-86e1e77a.pth'
    cfg = Config.fromfile(config)

    init_default_scope(cfg.default_scope)
    pipeline = Compose(cfg.val_pipeline)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MODELS.build(cfg.model)
    model.cfg = cfg

    # Infer
    checkpoint = _load_checkpoint(weights, map_location='cpu')
    _load_checkpoint_to_model(model, checkpoint)
    model.to(device)
    model.eval()

mising_data = []
with open('NTU_RGBD120_samples_with_missing_skeletons.txt', "r") as file:
    for line in file:
        line = line.strip()
        mising_data.append(line)

pickle_file = '../data/skeleton/ntu120_2d.pkl'
print(f"Loading {pickle_file} into memory...........")
start = time.time()
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)
print(f'Took {time.time() - start: 0.2f}s')

if flag:
    SAVE_DIR = '../data/STPTv2/examples'
else:
    SAVE_DIR = '../data/STPT/examples'
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
        if flag:
            generate_frame_plot_stgcn(cfg, model, device, annotation, output_path)
        else:
            generate_frame_plot(annotation, output_path)
        generate_kp_video(annotation, output_video)
        bar.update(1)
        i += 1
    if i == 120:
        break
bar.close()
