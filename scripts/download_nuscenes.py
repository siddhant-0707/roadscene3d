"""Download nuScenes mini dataset.

nuScenes mini is a small subset (~1.5 GB) perfect for development and testing.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_nuscenes_devkit():
    """Check if nuscenes-devkit is installed."""
    try:
        import nuscenes
        return True
    except ImportError:
        return False


def download_nuscenes_mini(output_dir: Path):
    """
    Download nuScenes mini dataset using nuscenes-devkit.
    
    Args:
        output_dir: Directory to save dataset
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading nuScenes mini dataset...")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        from nuscenes.nuscenes import NuScenes
        
        # nuScenes mini version
        version = 'v1.0-mini'
        dataroot = str(output_dir)
        
        logger.info(f"Initializing nuScenes {version}...")
        logger.info("Note: This will download ~1.5 GB of data.")
        logger.info("You may need to create an account at https://www.nuscenes.org/")
        
        # This will trigger download if not present
        nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        
        logger.info(f"Dataset downloaded successfully to {output_dir}")
        logger.info(f"Available scenes: {len(nusc.scene)}")
        
        return True
        
    except ImportError:
        logger.error("nuscenes-devkit not installed!")
        logger.info("Install with: pip install nuscenes-devkit")
        return False
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        logger.info("\nManual download instructions:")
        logger.info("1. Sign up at https://www.nuscenes.org/signup")
        logger.info("2. Download nuScenes mini from https://www.nuscenes.org/download")
        logger.info("3. Extract to the output directory")
        return False


def download_manual_instructions(output_dir: Path):
    """Print manual download instructions."""
    logger.info("\n" + "="*60)
    logger.info("Manual Download Instructions for nuScenes Mini")
    logger.info("="*60)
    logger.info("\n1. Sign up for nuScenes:")
    logger.info("   https://www.nuscenes.org/signup")
    logger.info("\n2. Download nuScenes mini:")
    logger.info("   https://www.nuscenes.org/download")
    logger.info("\n3. Extract files to:")
    logger.info(f"   {output_dir}")
    logger.info("\nRequired files:")
    logger.info("   - v1.0-mini.tgz (or extracted folder)")
    logger.info("   - metadata (scene, sample, etc.)")
    logger.info("\nExpected structure:")
    logger.info(f"   {output_dir}/")
    logger.info("   ├── v1.0-mini/")
    logger.info("   │   ├── maps/")
    logger.info("   │   ├── samples/")
    logger.info("   │   └── sweeps/")
    logger.info("\n4. After extraction, verify with:")
    logger.info("   python -c 'from nuscenes.nuscenes import NuScenes; nusc = NuScenes(version=\"v1.0-mini\", dataroot=\"data/nuscenes\"); print(f\"Scenes: {len(nusc.scene)}\")'")
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Download nuScenes mini dataset"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/nuscenes',
        help='Output directory for dataset'
    )
    parser.add_argument(
        '--manual',
        action='store_true',
        help='Show manual download instructions'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if args.manual:
        download_manual_instructions(output_dir)
        return 0
    
    # Check if devkit is installed
    if not check_nuscenes_devkit():
        logger.warning("nuscenes-devkit not installed. Showing manual instructions...")
        logger.info("\nTo install nuscenes-devkit:")
        logger.info("  pip install nuscenes-devkit")
        logger.info("\nOr use manual download:")
        download_manual_instructions(output_dir)
        return 1
    
    # Try automatic download
    success = download_nuscenes_mini(output_dir)
    
    if not success:
        download_manual_instructions(output_dir)
        return 1
    
    logger.info("\n✅ Download complete!")
    logger.info(f"\nNext steps:")
    logger.info("1. Verify dataset: python -c 'from nuscenes.nuscenes import NuScenes; nusc = NuScenes(version=\"v1.0-mini\", dataroot=\"{}\"); print(f\"Scenes: {{len(nusc.scene)}}\")'".format(output_dir))
    logger.info("2. Update data loader for nuScenes format")
    logger.info("3. Start training!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
