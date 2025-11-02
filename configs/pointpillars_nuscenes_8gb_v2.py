"""PointPillars configuration for nuScenes mini, optimized for 8GB VRAM.

Based on MMDetection3D's official PointPillars config.
Optimizations for 8GB VRAM:
- Batch size 2 (reduced from 4)
- Mixed precision training (FP16)
- Reduced workers
"""

_base_ = [
    '../mmdetection3d/configs/_base_/models/pointpillars_hv_secfpn_nuscenes.py',
    '../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../mmdetection3d/configs/_base_/schedules/schedule-2x.py',
    '../mmdetection3d/configs/_base_/default_runtime.py'
]

# Override dataset settings
data_root = 'data/nuscenes/v1.0-mini/'
train_ann_file = data_root + 'nuscenes_infos_train.pkl'
val_ann_file = data_root + 'nuscenes_infos_val.pkl'

# Reduce batch size for 8GB VRAM
train_dataloader = dict(
    batch_size=2,  # Reduced from 4
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        ann_file=train_ann_file))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        ann_file=val_ann_file))

test_dataloader = val_dataloader

# Update evaluator data paths
val_evaluator = dict(
    data_root=data_root,
    ann_file=val_ann_file)
test_evaluator = val_evaluator

# Training schedule - shorter for mini dataset
train_cfg = dict(max_epochs=40, val_interval=5)  # Reduced from 80 epochs

# Enable mixed precision training
optim_wrapper = dict(
    type='AmpOptimWrapper',  # Use AMP for FP16
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

# Reduce checkpoint frequency to save disk space
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=2))  # Only keep 2 latest checkpoints

