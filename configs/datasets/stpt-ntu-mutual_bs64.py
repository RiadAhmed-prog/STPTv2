CLASSES = (
        'punch/slap', 'kicking', 'pushing', 'pat on back', 'point finger', 'hugging', 'giving object', 'touch pocket',
        'shaking hands', 'walking towards', 'walking apart', 'hit with object', 'wield knife', 'knock over', 'grab stuff',
        'shoot with gun', 'step on foot', 'high-five', 'cheers and drink', 'carry object', 'take a photo', 'follow',
        'whisper', 'exchange things', 'support somebody', 'rock-paper-scissors'
)

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

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/STPT/mutual_actions',
        split='train',
        metainfo={'classes': CLASSES},
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/STPT/mutual_actions',
        split='val',
        metainfo={'classes': CLASSES},
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = dict(
    batch_size=64,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/STPT/mutual_actions/',
        data_prefix='test/',
        ann_file='meta/test.txt',
        metainfo={'classes': CLASSES},
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = [
    dict(type='Accuracy', topk=(1, 5)),
    dict(type='SingleLabelMetric', items=['precision', 'recall']),
    dict(type='ConfusionMatrix')
]
