"""Verify nuScenes dataset setup and data loader."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.nuscenes_loader import NuScenesDataset
import numpy as np

def main():
    print("=" * 60)
    print("nuScenes Dataset Verification")
    print("=" * 60)
    print()
    
    # Test 1: Dataset initialization
    print("1. Testing dataset initialization...")
    try:
        dataset = NuScenesDataset(
            data_root='data/nuscenes/v1.0-mini',
            version='v1.0-mini',
            modality={'use_lidar': True, 'use_camera': False},
            filter_empty_gt=False
        )
        print(f"   ✅ Dataset initialized: {len(dataset)} samples")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return 1
    
    # Test 2: Sample loading
    print("\n2. Testing sample loading...")
    try:
        sample = dataset[0]
        print(f"   ✅ Sample 0 loaded")
        print(f"      Points shape: {sample['points'].shape}")
        print(f"      GT boxes: {sample['gt_bboxes_3d'].shape[0]} boxes")
        print(f"      GT labels: {sample['gt_labels_3d'].shape}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return 1
    
    # Test 3: Find samples with annotations
    print("\n3. Testing annotation extraction...")
    try:
        samples_with_annotations = 0
        total_boxes = 0
        class_counts = {}
        
        for i in range(min(50, len(dataset))):
            sample = dataset[i]
            n_boxes = sample['gt_bboxes_3d'].shape[0]
            if n_boxes > 0:
                samples_with_annotations += 1
                total_boxes += n_boxes
                for label in sample['gt_labels_3d']:
                    class_name = dataset.CLASSES[label]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"   ✅ Annotations extracted")
        print(f"      Samples with annotations: {samples_with_annotations}/50")
        print(f"      Total boxes: {total_boxes}")
        print(f"      Class distribution:")
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"         {cls}: {count}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return 1
    
    # Test 4: Point cloud format
    print("\n4. Testing point cloud format...")
    try:
        sample = dataset[0]
        points = sample['points']
        assert points.shape[1] == 4, f"Expected 4 channels, got {points.shape[1]}"
        assert points.dtype == np.float32, f"Expected float32, got {points.dtype}"
        print(f"   ✅ Point cloud format correct")
        print(f"      Shape: {points.shape}")
        print(f"      Dtype: {points.dtype}")
        print(f"      Range: [{points[:, 0].min():.1f}, {points[:, 0].max():.1f}] x [{points[:, 1].min():.1f}, {points[:, 1].max():.1f}]")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return 1
    
    # Test 5: Box format
    print("\n5. Testing bounding box format...")
    try:
        sample_with_boxes = None
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample['gt_bboxes_3d'].shape[0] > 0:
                sample_with_boxes = sample
                break
        
        if sample_with_boxes:
            boxes = sample_with_boxes['gt_bboxes_3d']
            assert boxes.shape[1] == 7, f"Expected 7 box dimensions, got {boxes.shape[1]}"
            print(f"   ✅ Bounding box format correct")
            print(f"      Shape: {boxes.shape}")
            print(f"      Format: [x, y, z, l, w, h, yaw]")
            print(f"      First box: {boxes[0]}")
        else:
            print(f"   ⚠️  No samples with boxes found")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! nuScenes setup is working correctly.")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Install MMDetection3D when ready for training")
    print("2. Use config: configs/pointpillars_nuscenes_8gb.py")
    print("3. Start training: python src/training/train.py --config configs/pointpillars_nuscenes_8gb.py")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
