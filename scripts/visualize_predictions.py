#!/usr/bin/env python3
"""Visualize model predictions on point clouds."""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "mmdetection3d"))

from mmengine.config import Config
from mmengine.runner import Runner
from src.visualization.pointcloud_viz import PointCloudVisualizer


def main():
    parser = argparse.ArgumentParser(description="Visualize predictions on point clouds")
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index to visualize")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--save-path", type=str, default=None, help="Path to save visualization")
    parser.add_argument("--show", action="store_true", help="Show interactive visualization")
    
    args = parser.parse_args()
    
    # Load config
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    
    # Set work_dir if not present (required by Runner)
    # Check if work_dir exists in config dict, not as attribute
    if 'work_dir' not in cfg:
        cfg.work_dir = str(Path(args.checkpoint).parent)
    elif cfg.work_dir is None:
        cfg.work_dir = str(Path(args.checkpoint).parent)
    
    # Create runner
    runner = Runner.from_cfg(cfg)
    
    # Monkey-patch torch.load to use weights_only=False for our trusted checkpoint
    # PyTorch 2.6+ defaults to weights_only=True which blocks MMDetection3D checkpoints
    # This is safe since we trust our own trained checkpoints
    import torch
    original_torch_load = torch.load
    
    def torch_load_with_trust(*args, **kwargs):
        """Wrapper for torch.load that sets weights_only=False."""
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    
    torch.load = torch_load_with_trust
    
    try:
        runner.load_checkpoint(args.checkpoint)
    finally:
        # Restore original torch.load
        torch.load = original_torch_load
    
    # Get dataset
    if args.split == "val":
        dataset = runner.build_dataloader(cfg.val_dataloader).dataset
    elif args.split == "test":
        dataset = runner.build_dataloader(cfg.test_dataloader).dataset
    else:
        dataset = runner.build_dataloader(cfg.train_dataloader).dataset
    
    # Get sample
    if args.sample_idx >= len(dataset):
        print(f"Error: Sample index {args.sample_idx} out of range (dataset has {len(dataset)} samples)")
        return
    
    sample = dataset.get_data_info(args.sample_idx)
    
    # Load point cloud
    lidar_path = sample['lidar_points']['lidar_path']
    # Check if path is already absolute or includes data_root
    if Path(lidar_path).is_absolute() or str(lidar_path).startswith('data/'):
        points_path = Path(lidar_path)
    else:
        points_path = Path(cfg.data_root) / lidar_path
    
    print(f"Loading point cloud from: {points_path}")
    points = np.fromfile(str(points_path), dtype=np.float32).reshape(-1, 5)  # x, y, z, intensity, ring
    
    # Get ground truth boxes
    gt_boxes = []
    if 'instances' in sample:
        for instance in sample['instances']:
            gt_boxes.append({
                'center': instance['bbox_3d'][:3],
                'size': instance['bbox_3d'][3:6],
                'rotation': instance['bbox_3d'][6] if len(instance['bbox_3d']) >= 7 else 0.0,
                'label': instance['bbox_label']
            })
    
    # Run inference
    model = runner.model
    model.eval()
    
    with torch.no_grad():
        # Prepare input (simplified - would need proper pipeline)
        # results = model.test_step([sample])
        # predictions = results[0]
        
        # For now, create placeholder predictions
        pred_boxes = []
    
    # Visualize
    visualizer = PointCloudVisualizer()
    visualizer.visualize_scene(
        points=points[:, :4],  # x, y, z, intensity
        gt_boxes=gt_boxes,
        pred_boxes=pred_boxes,
        save_path=args.save_path,
        show=args.show
    )
    
    print(f"✅ Visualization complete!")
    if args.save_path:
        print(f"📁 Saved to: {args.save_path}")


if __name__ == "__main__":
    main()

