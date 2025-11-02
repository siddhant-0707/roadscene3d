"""PointPillars configuration for nuScenes dataset, optimized for 8GB VRAM.

This config uses:
- Batch size 2 (can be adjusted)
- Gradient accumulation (effective batch size 8)
- Mixed precision training (FP16)
- Optimized voxelization settings for nuScenes
"""

# Custom imports to register MMDetection3D components
custom_imports = dict(
    imports=['mmdet3d.datasets', 'mmdet3d.models', 'mmdet3d.evaluation'],
    allow_failed_imports=False)

# Model configuration
model = dict(
    type='PointPillars',
    voxel_layer=dict(
        max_num_points=32,
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],  # nuScenes standard range
        voxel_size=[0.1, 0.1, 0.2],  # Standard nuScenes voxel size
        max_voxels=(16000, 40000)  # Reduced for 8GB VRAM
    ),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=(0.1, 0.1, 0.2),
        point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    ),
    middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=(512, 512),  # Adjusted for nuScenes range
    ),
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256],
    ),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128],
    ),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=10,  # nuScenes has 10 classes
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            # nuScenes anchor ranges - adjusted for 10 classes
            ranges=[
                [-51.2, -51.2, -1.8, 51.2, 51.2, -1.8],  # car
                [-51.2, -51.2, -1.9, 51.2, 51.2, -1.9],  # truck
                [-51.2, -51.2, -1.9, 51.2, 51.2, -1.9],  # bus
                [-51.2, -51.2, -1.9, 51.2, 51.2, -1.9],  # trailer
                [-51.2, -51.2, -1.8, 51.2, 51.2, -1.8],  # pedestrian
                [-51.2, -51.2, -1.8, 51.2, 51.2, -1.8],  # bicycle
                [-51.2, -51.2, -1.8, 51.2, 51.2, -1.8],  # motorcycle
                [-51.2, -51.2, -2.0, 51.2, 51.2, -2.0],  # traffic_cone
                [-51.2, -51.2, -2.0, 51.2, 51.2, -2.0],  # barrier
                [-51.2, -51.2, -1.8, 51.2, 51.2, -1.8],  # bicycle_rack
            ],
            # Anchor sizes for each class (l, w, h)
            sizes=[
                [4.6, 1.9, 1.8],  # car
                [10.6, 2.9, 3.2],  # truck
                [12.4, 2.9, 3.5],  # bus
                [10.9, 2.9, 3.5],  # trailer
                [0.8, 0.6, 1.8],  # pedestrian
                [1.9, 0.7, 1.8],  # bicycle
                [2.0, 0.9, 1.8],  # motorcycle
                [0.6, 0.6, 2.0],  # traffic_cone
                [0.6, 0.8, 1.6],  # barrier
                [1.5, 0.5, 1.5],  # bicycle_rack
            ],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=1000,
        max_num=500))

# Training configuration
train_batch_size = 2  # Reduced for 8GB VRAM
gradient_accumulation_steps = 4  # Effective batch size = 2 * 4 = 8
use_amp = True  # Mixed precision training (required for 8GB VRAM)

# Dataset configuration
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/v1.0-mini'
ann_file_train = 'nuscenes_infos_train.pkl'
ann_file_val = 'nuscenes_infos_val.pkl'
data_prefix = dict(pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP')

# nuScenes classes (10 official classes)
class_names = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']

# Data pipeline
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='ObjectSample',
        db_sampler=dict(
            data_root=data_root,
            info_path=data_root + '/dbinfos_train.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(car=5, truck=5, bus=5, trailer=5,
                                        construction_vehicle=5, pedestrian=5, bicycle=5,
                                        motorcycle=5, traffic_cone=3, barrier=3)),
            classes=class_names,
            sample_groups=dict(car=15, truck=15, bus=15, trailer=15,
                              construction_vehicle=15, pedestrian=15, bicycle=15,
                              motorcycle=15, traffic_cone=10, barrier=10))),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.2, 0.2, 0.2]),
    dict(type='PointsRangeFilter', point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(type='ObjectRangeFilter', point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='GlobalRotScaleTrans', rot_range=[0, 0], scale_ratio_range=[1., 1.],
                 translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(type='DefaultFormatBundle3D', class_names=class_names),
            dict(type='Collect3D', keys=['points'])
        ])
]

# Data loaders (new MMEngine format)
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_train,
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=False,
        box_type_3d='LiDAR',
        filter_empty_gt=True))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_val,
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True,
        box_type_3d='LiDAR'))

test_dataloader = val_dataloader

# Evaluators
val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + '/' + ann_file_val,
    metric='bbox')
test_evaluator = val_evaluator

# Optimizer wrapper (new MMEngine format)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

# Learning rate scheduler
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=80,
        eta_min=1e-5,
        begin=0,
        end=80,
        by_epoch=True)
]

# Training configuration
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

# Environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
