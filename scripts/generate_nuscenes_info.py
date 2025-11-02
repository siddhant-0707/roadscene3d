#!/usr/bin/env python3
"""Generate nuScenes info files for MMDetection3D.

This script uses MMDetection3D's converter to generate the required
.pkl info files from the raw nuScenes dataset.
"""

import sys
from pathlib import Path

# Add mmdetection3d to path
mmdet3d_path = Path(__file__).parent.parent / 'mmdetection3d'
sys.path.insert(0, str(mmdet3d_path))

from tools.dataset_converters.nuscenes_converter import create_nuscenes_infos


def main():
    """Generate nuScenes info files."""
    data_root = 'data/nuscenes/v1.0-mini'
    version = 'v1.0-mini'
    info_prefix = 'nuscenes'
    
    print("=" * 60)
    print("Generating nuScenes info files for MMDetection3D")
    print("=" * 60)
    print(f"Data root: {data_root}")
    print(f"Version: {version}")
    print(f"Info prefix: {info_prefix}")
    print()
    
    try:
        create_nuscenes_infos(
            root_path=data_root,
            info_prefix=info_prefix,
            version=version,
            max_sweeps=10
        )
        
        print()
        print("=" * 60)
        print("✅ Successfully generated info files!")
        print("=" * 60)
        print(f"Train info: {data_root}/nuscenes_infos_train.pkl")
        print(f"Val info: {data_root}/nuscenes_infos_val.pkl")
        print()
        print("You can now use these info files in your MMDetection3D config.")
        
    except Exception as e:
        print(f"❌ Error generating info files: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

