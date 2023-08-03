_base_ = [
    'datasets/stpt-ntu-medical_bs64.py',
    'schedules/stpt_bs64.py',
    'default_runtime.py',
]
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNetV2', arch='b3'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=12,
        in_channels=1536,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# dataset settings
dataset_type = 'ImageNet'
data_preprocessor = dict(
    num_classes=12,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetRandomCrop', scale=240),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetCenterCrop', crop_size=300, crop_padding=0),
    dict(type='PackInputs'),
]

train_dataloader = dict(batch_size=16,
                        dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=16,
                      dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(batch_size=16,
                       dataset=dict(pipeline=test_pipeline))
