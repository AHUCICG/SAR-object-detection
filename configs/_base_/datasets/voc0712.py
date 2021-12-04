# dataset settings
dataset_type = 'VOCDataset'
# data_root = '/home/ahu-xiarunfan/下载/Swin-Transformer-Object-Detection-master/data/SSDD/'
data_root = '/home/ahu-xiarunfan/下载/DIOR/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    # mean=[0.1227, 0.1227, 0.1227], std=[0.1049, 0.1049, 0.1048], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640,640), keep_ratio=640),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),


]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(800, 800),
        img_scale=[(640, 640)],

        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                # data_root + 'VOC2007/ImageSets/Main/train.txt'
                # data_root + 'VOC2012/ImageSets/Main/train.txt'
                data_root + 'VOC2007/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[data_root + 'VOC2007/'],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'VOC2007/ImageSets/Main/val.txt',
        ann_file=data_root + 'VOC2007/ImageSets/Main/vall.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        # img_prefix=data_root + 'VOC2007/',
        ann_file=data_root + 'VOC2007/ImageSets/Main/vall.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
