# Installation

---

After cloning the repo:\
`cd STPTv2`\
`pip install virtualenv` (if you don't already have virtualenv installed)\
`python3 -m virtualenv envstpt` to create the virtual environment for the project\
`source envstpt/bin/activate` to activate virtual environment\
Use `pip install -r requirements.txt` to install required dependencies

Firstly, install PyTorch (<2.0.0) following [official instructions](https://pytorch.org/get-started/locally/)\
Now, install mmdpretrain and mmaction2 by following:\
`mim install mmcv`
1. mmpretrain
````
git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
mim install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
````
2. mmaction2
````
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
mim install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
````
# Instructions

---
Download the `ntu120_2d.pkl` file containing skeleton information of the NTU120-RGB dataset from this [link](https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_2d.pkl) and extract to `data/skeleton` folder \
Use the following command to generate STPT images of NTU120-RGB dataset:
```python
python generate_stpt_dataset {dataset} --[OPTIONAL]
```
**dataset**: Must be either 'NTU60' or 'NTU120'\
[OPTIONAL] args:\
**group**: must be one of ['medical_conditions', 'daily_actions', 'mutual_actions']\
**test**: if set, will generate 'test' split\
**stgcn-config**: config file for the stgcn/stgcn++ model for important keypoint inference\
**stgcn-weights**: stgcn/stgcn++ model weights 

Dataset will be generated in the _**data**_ directory in the following structure:
```
stpt-activity
├── data
│   └── skeleton
│           └── ntu60_2d.pkl
│           └── ...
│   └── STPT
│        └── medical_conditions
│           ├── meta
│           ├── test
│           ├── train
│           └── val
└── ...
```
Run `./train.sh` to train a model from the _**configs**_ directory

(if permission denied error shows up use the command `chmod +x ./train.sh`)

Run `./test.sh` to test the model from the _**configs**_ directory, with loaded checkpoint from the **_work_dirs_** directory.\

**N.B.**\
If `--gsheets` argument is used when running `./test.sh`, results on the **test** set will be saved to the [Google Sheets file](https://docs.google.com/spreadsheets/d/1IQAqw0tQ5ySzbnyKn2i86hgak1PZwmuOFrYzbvUGcrI/edit?usp=sharing)
