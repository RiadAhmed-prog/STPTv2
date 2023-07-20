CLASSES = (
        'drink water', 'eat meal', 'brush teeth', 'brush hair', 'drop', 'pick up', 'throw', 'sit down', 'stand up',
        'clapping', 'reading', 'writing', 'tear up paper', 'put on jacket', 'take off jacket', 'put on a shoe',
        'take off a shoe', 'put on glasses', 'take off glasses', 'put on a hat/cap', 'take off a hat/cap', 'cheer up',
        'hand waving', 'kicking something', 'reach into pocket', 'hopping', 'jump up', 'phone call',
        'play with phone/tablet', 'type on a keyboard', 'point to something', 'taking a selfie',
        'check time (from watch)',
        'rub two hands', 'nod head/bow', 'shake head', 'wipe face', 'salute', 'put palms together',
        'cross hands in front',
        'put on headphone', 'take off headphone', 'shoot at basket', 'bounce ball', 'tennis bat swing',
        'juggle table tennis ball', 'hush', 'flick hair', 'thumb up', 'thumb down', 'make OK sign', 'make victory sign',
        'staple book', 'counting money', 'cutting nails', 'cutting paper', 'snap fingers', 'open bottle', 'sniff/smell',
        'squat down', 'toss a coin', 'fold paper', 'ball up paper', 'play magic cube', 'apply cream on face',
        'apply cream on hand', 'put on bag', 'take off bag', 'put object into bag', 'take object out of bag',
        'open a box',
        'move heavy objects', 'shake fist', 'throw up cap/hat', 'capitulate', 'cross arms', 'arm circles', 'arm swings',
        'run on the spot', 'butt kicks', 'cross toe touch', 'side kick'
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
        data_root='data/STPT/daily_actions',
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
        data_root='data/STPT/daily_actions',
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
        data_root='data/STPT/daily_actions/',
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
