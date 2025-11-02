"""Download Waymo Open Dataset Perception v2.0.1 from GCP bucket.

This script downloads the Waymo Perception Dataset v2.0.1 in modular format,
allowing selective download of camera, LiDAR, and label components.
"""

import argparse
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
import concurrent.futures

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Waymo Open Dataset buckets
# Note: Bucket structure may vary. Check with: gcloud storage ls gs://waymo_open_dataset_v_*
WAYMO_BUCKET_V2 = "gs://waymo_open_dataset_v_2_0_1_individual_files"  # Modular v2.0.1
WAYMO_BUCKET_V1 = "gs://waymo_open_dataset_v_1_4_3"  # v1.4.3 with maps

# Common bucket paths (may need adjustment based on actual structure)
# For v1.4.3, files are in individual_files/{split}/
# Check structure first with: gcloud storage ls gs://waymo_open_dataset_v_1_4_3/individual_files/
COMPONENTS_V1 = {
    'training': 'individual_files/training/*.tfrecord',
    'validation': 'individual_files/validation/*.tfrecord',
    'testing': 'individual_files/testing/*.tfrecord',
}

# For v2.0.1 modular format (structure TBD - user should check bucket first)
COMPONENTS_V2 = {
    'camera_training': 'perception_2_0_1/camera_training/*.tfrecord',
    'lidar_training': 'perception_2_0_1/lidar_training/*.tfrecord',
    'labels_3d_training': 'perception_2_0_1/labels_3d_training/*.tfrecord',
    # Add validation and testing as needed
}


def check_gcloud_auth():
    """Check if gcloud is authenticated."""
    try:
        result = subprocess.run(
            ['gcloud', 'auth', 'list', '--filter=status:ACTIVE', '--format=value(account)'],
            capture_output=True,
            text=True,
            check=True
        )
        if result.stdout.strip():
            logger.info(f"Authenticated as: {result.stdout.strip()}")
            return True
        else:
            logger.error("Not authenticated with gcloud. Run: gcloud auth login")
            return False
    except subprocess.CalledProcessError:
        logger.error("gcloud CLI not found or not working. Please install Google Cloud SDK.")
        return False
    except FileNotFoundError:
        logger.error("gcloud CLI not found. Please install Google Cloud SDK.")
        return False


def list_bucket_files(bucket: str, prefix: str = "") -> List[str]:
    """
    List files in GCP bucket.
    
    Args:
        bucket: GCS bucket path (e.g., gs://bucket)
        prefix: Prefix to filter files (e.g., 'perception/training/')
        
    Returns:
        List of file paths (full gs:// URLs)
    """
    try:
        # Use gcloud storage (newer, recommended)
        cmd = ['gcloud', 'storage', 'ls', '-r']
        if prefix:
            cmd.append(f'{bucket}/{prefix}')
        else:
            cmd.append(f'{bucket}/**/*.tfrecord')
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Filter for .tfrecord files only, exclude directories (those ending with /)
        files = [
            line.strip() for line in result.stdout.split('\n') 
            if line.strip() and line.endswith('.tfrecord') and not line.strip().endswith('/')
        ]
        return files
    except subprocess.CalledProcessError as e:
        # Fallback to gsutil
        try:
            logger.info("Trying gsutil as fallback...")
            cmd = ['gsutil', '-m', 'ls']
            if prefix:
                cmd.append(f'{bucket}/{prefix}**/*.tfrecord')
            else:
                cmd.append(f'{bucket}/**/*.tfrecord')
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            files = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            return files
        except Exception as err:
            logger.error(f"Failed to list files: {e}. Also tried gsutil: {err}")
            logger.info(f"Try manually checking: gcloud storage ls {bucket}/")
            return []


def download_file(gcs_path: str, local_path: Path, use_parallel: bool = True):
    """
    Download a single file from GCS.
    
    Args:
        gcs_path: GCS file path
        local_path: Local destination path
        use_parallel: Use parallel composite uploads
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try gcloud storage (newer, faster)
        cmd = ['gcloud', 'storage', 'cp']
        if use_parallel:
            cmd.append('-m')  # Use parallel composite uploads
        cmd.extend([gcs_path, str(local_path)])
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        # Fallback to gsutil
        try:
            cmd = ['gsutil', '-m', 'cp', gcs_path, str(local_path)]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except Exception as e:
            logger.error(f"Failed to download {gcs_path}: {e}")
            return False


def download_files_parallel(file_pairs: List[tuple], max_workers: int = 4):
    """
    Download multiple files in parallel.
    
    Args:
        file_pairs: List of (gcs_path, local_path) tuples
        max_workers: Maximum parallel downloads
    """
    def download_wrapper(pair):
        gcs_path, local_path = pair
        return download_file(gcs_path, local_path)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_wrapper, pair) for pair in file_pairs]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading"):
            future.result()


def download_component(
    component: str,
    split: str,
    output_dir: Path,
    bucket: str = WAYMO_BUCKET_V1,
    max_files: Optional[int] = None
) -> List[Path]:
    """
    Download a component (camera, lidar, labels, etc.) for a split.
    
    Args:
        component: Component name ('camera', 'lidar', 'labels', '3d_labels')
        split: Dataset split ('training', 'validation', 'testing')
        output_dir: Output directory
        bucket: GCS bucket path
        max_files: Maximum number of files to download (for subset)
        
    Returns:
        List of downloaded file paths
    """
    if split not in ['training', 'validation', 'testing']:
        logger.error(f"Unknown split: {split}")
        return []
    
    # For v1.4.3, files are in individual_files/{split}/
    # For v2.0.1, structure may differ - user needs to check
    if 'v_1_4' in bucket or 'v_2_0_1' not in bucket:
        # Use v1.4.3 structure - files are in individual_files/{split}/
        prefix = f'individual_files/{split}/'
    else:
        # For v2.0.1 modular, structure depends on component
        prefix = f'perception_2_0_1/{component}_{split}/'
    
    logger.info(f"Listing files for {split} split in {bucket}...")
    logger.info(f"Using prefix: {prefix}")
    
    files = list_bucket_files(bucket, prefix)
    
    if not files:
        logger.warning(f"No files found with prefix {prefix}")
        logger.info("Trying alternative: listing all .tfrecord files...")
        all_files = list_bucket_files(bucket)
        logger.info(f"Found {len(all_files)} total .tfrecord files in bucket")
        if all_files:
            logger.info("First few files:")
            for f in all_files[:5]:
                logger.info(f"  {f}")
        logger.error(f"Could not find files with expected prefix. Please check bucket structure manually.")
        logger.info(f"Try: gcloud storage ls {bucket}/")
        return []
    
    logger.info(f"Found {len(files)} files for {split} split")
    
    # Limit files if requested
    if max_files and len(files) > max_files:
        logger.info(f"Limiting to {max_files} files (found {len(files)} total)")
        files = files[:max_files]
    
    # Prepare download paths
    # For v1.4.3, files contain all components, so we don't separate by component
    # For v2.0.1 modular, separate by component
    if 'v_2_0_1' in bucket and component:
        output_component_dir = output_dir / component / split
    else:
        output_component_dir = output_dir / split
    
    file_pairs = []
    
    for file_url in files:
        # Extract filename from URL
        filename = Path(file_url).name
        local_path = output_component_dir / filename
        file_pairs.append((file_url, local_path))
    
    logger.info(f"Downloading {len(file_pairs)} files to {output_component_dir}...")
    download_files_parallel(file_pairs, max_workers=4)
    
    # Return downloaded paths
    downloaded = [local_path for _, local_path in file_pairs if local_path.exists()]
    logger.info(f"Successfully downloaded {len(downloaded)}/{len(file_pairs)} files")
    
    return downloaded


def download_subset(
    output_dir: Path,
    components: List[str] = ['lidar', '3d_labels'],
    n_train: int = 120,
    n_val: int = 25,
    n_test: int = 25,
    bucket: str = WAYMO_BUCKET_V1
):
    """
    Download a subset of Waymo dataset for training.
    
    Args:
        output_dir: Output directory
        components: Components to download
        n_train: Number of training files
        n_val: Number of validation files
        n_test: Number of test files
        bucket: GCS bucket path
    """
    logger.info(f"Downloading Waymo subset:")
    logger.info(f"  Components: {components} (for v2.0.1 modular only)")
    logger.info(f"  Training: {n_train} files")
    logger.info(f"  Validation: {n_val} files")
    logger.info(f"  Testing: {n_test} files")
    logger.info(f"  Bucket: {bucket}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_files = {}
    
    # For v1.4.3, files contain all components, so component parameter is ignored
    # For v2.0.1, download each component separately
    if 'v_2_0_1' in bucket:
        # Modular format - download each component
        for component in components:
            for split, n_files in [('training', n_train), ('validation', n_val), ('testing', n_test)]:
                files = download_component(component, split, output_dir, bucket, max_files=n_files)
                downloaded_files[f"{component}/{split}"] = files
    else:
        # v1.4.3 format - files contain everything, component is ignored
        for split, n_files in [('training', n_train), ('validation', n_val), ('testing', n_test)]:
            files = download_component('all', split, output_dir, bucket, max_files=n_files)
            downloaded_files[f"{split}"] = files
    
    # Save download manifest
    manifest_path = output_dir / 'download_manifest.json'
    manifest = {
        'components': components,
        'splits': {
            'training': n_train,
            'validation': n_val,
            'testing': n_test
        },
        'files': {k: [str(f) for f in v] for k, v in downloaded_files.items()}
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Download manifest saved to {manifest_path}")
    logger.info("Download complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Download Waymo Open Dataset Perception v2.0.1"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/waymo',
        help='Output directory for downloaded files'
    )
    parser.add_argument(
        '--components',
        nargs='+',
        choices=['camera', 'lidar', 'labels', '3d_labels'],
        default=['lidar', '3d_labels'],
        help='Components to download'
    )
    parser.add_argument(
        '--n-train',
        type=int,
        default=120,
        help='Number of training files to download'
    )
    parser.add_argument(
        '--n-val',
        type=int,
        default=25,
        help='Number of validation files to download'
    )
    parser.add_argument(
        '--n-test',
        type=int,
        default=25,
        help='Number of test files to download'
    )
    parser.add_argument(
        '--bucket',
        type=str,
        default=WAYMO_BUCKET_V1,
        help='GCS bucket path (default: v1.4.3). For v2.0.1 use: gs://waymo_open_dataset_v_2_0_1_individual_files'
    )
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='Only list available files, do not download'
    )
    
    args = parser.parse_args()
    
    # Check authentication
    if not check_gcloud_auth():
        return 1
    
    output_dir = Path(args.output_dir)
    
    if args.list_only:
        logger.info("Listing available files (not downloading)...")
        logger.info(f"Bucket: {args.bucket}")
        
        # List top-level structure
        logger.info("\nListing bucket structure...")
        try:
            cmd = ['gcloud', 'storage', 'ls', args.bucket]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Top-level directories:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
        except:
            pass
        
        # Try to find .tfrecord files
        files = list_bucket_files(args.bucket)
        logger.info(f"\nFound {len(files)} .tfrecord files total")
        
        if files:
            logger.info("\nSample files:")
            for f in files[:10]:
                logger.info(f"  {f}")
        return 0
    
    # Download subset
    download_subset(
        output_dir=output_dir,
        components=args.components,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        bucket=args.bucket
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
