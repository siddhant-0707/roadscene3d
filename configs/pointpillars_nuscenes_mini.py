"""PointPillars for nuScenes mini dataset, optimized for 8GB VRAM."""

_base_ = [
    '../mmdetection3d/configs/_base_/models/pointpillars_hv_fpn_nus.py',
    '../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../mmdetection3d/configs/_base_/schedules/schedule-2x.py',
    '../mmdetection3d/configs/_base_/default_runtime.py'
]

# Override data root for mini dataset
data_root = 'data/nuscenes/v1.0-mini/'

# Add version to metainfo for mini dataset
metainfo = dict(
    classes=['car', 'truck', 'trailer', 'bus', 'construction_vehicle',
             'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'],
    version='v1.0-mini')

# Reduce batch size for 8GB VRAM
train_dataloader = dict(
    batch_size=1,  # Further reduced to 1 for 8GB VRAM
    num_workers=2,  # Reduced workers to save RAM
    dataset=dict(
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',
        metainfo=metainfo))  # Add metainfo with version

val_dataloader = dict(
    batch_size=1,
    num_workers=1,  # Reduced workers
    dataset=dict(
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        metainfo=metainfo))  # Add metainfo with version

test_dataloader = val_dataloader

# Update evaluator with correct paths and version for mini dataset
val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    metric='bbox',
    jsonfile_prefix='./work_dirs/nuscenes_mini_results')
test_evaluator = val_evaluator

# Shorter training for mini dataset (40 epochs instead of default 24)
train_cfg = dict(max_epochs=40, val_interval=5)

# Keep only 2 latest checkpoints to save disk space
default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=2))

