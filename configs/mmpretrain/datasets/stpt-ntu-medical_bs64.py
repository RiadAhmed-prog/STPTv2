CLASSES = (
    'sneeze/cough', 'staggering', 'falling down', 'headache', 'chest pain', 'back pain', 'neck pain',
    'nausea/vomiting', 'fan self', 'yawn', 'stretch oneself', 'blow nose'
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
        data_root='data/STPT-skeleton/medical_conditions',
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
        data_root='data/STPT-skeleton/medical_conditions',
        split='val',
        metainfo={'classes': CLASSES},
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_dataloader = val_dataloader

# # If you want standard test, please manually configure the test dataset
# test_dataloader = dict(
#     batch_size=64,
#     num_workers=2,
#     dataset=dict(
#         type=dataset_type,
#         data_root='data/STPTv2/medical_conditions/',
#         data_prefix='val/',
#         ann_file='meta/test.txt',
#         metainfo={'classes': CLASSES},
#         pipeline=test_pipeline),
#     sampler=dict(type='DefaultSampler', shuffle=False),
# )

test_evaluator = [
    dict(type='Accuracy', topk=(1, 5)),
    dict(type='SingleLabelMetric', items=['precision', 'recall']),
    dict(type='ConfusionMatrix')
]
