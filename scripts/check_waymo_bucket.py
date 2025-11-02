"""Quick script to check Waymo bucket structure and list available files.

This helps identify the correct bucket paths for downloading.
"""

import subprocess
import argparse
import sys

def check_bucket(bucket_path: str):
    """List contents of a GCP bucket."""
    print(f"\n{'='*60}")
    print(f"Checking bucket: {bucket_path}")
    print(f"{'='*60}\n")
    
    # Check if bucket exists and list top-level
    try:
        print("Top-level directories:")
        cmd = ['gcloud', 'storage', 'ls', bucket_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                print(f"  {line}")
    except subprocess.CalledProcessError as e:
        print(f"Error accessing bucket: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False
    
    # Try to find perception directory
    print("\nLooking for perception data...")
    perception_paths = [
        f"{bucket_path}/perception/",
        f"{bucket_path}/perception_2_0_1/",
    ]
    
    for path in perception_paths:
        try:
            cmd = ['gcloud', 'storage', 'ls', path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if result.stdout.strip():
                print(f"\nFound: {path}")
                print("Contents:")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        print(f"  {line}")
        except:
            pass
    
    # Count .tfrecord files
    print("\nCounting .tfrecord files...")
    try:
        cmd = ['gcloud', 'storage', 'ls', '-r', f'{bucket_path}/**/*.tfrecord']
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        count = len([l for l in result.stdout.strip().split('\n') if l.strip()])
        print(f"Total .tfrecord files: {count}")
        
        if count > 0:
            print("\nSample files (first 5):")
            for line in result.stdout.strip().split('\n')[:5]:
                if line.strip():
                    print(f"  {line}")
    except Exception as e:
        print(f"Could not count files: {e}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Check Waymo bucket structure"
    )
    parser.add_argument(
        '--bucket',
        type=str,
        default='gs://waymo_open_dataset_v_1_4_3',
        help='Bucket to check'
    )
    
    args = parser.parse_args()
    
    print("Waymo Open Dataset Bucket Checker")
    print("=" * 60)
    print("\nCommon buckets to check:")
    print("  - gs://waymo_open_dataset_v_1_4_3 (v1.4.3)")
    print("  - gs://waymo_open_dataset_v_2_0_1_individual_files (v2.0.1 modular)")
    print("  - gs://waymo_open_dataset_v_2_0_1 (v2.0.1)")
    print()
    
    success = check_bucket(args.bucket)
    
    if success:
        print("\n" + "="*60)
        print("SUCCESS: Bucket accessible!")
        print("\nNext steps:")
        print("1. Note the structure above")
        print("2. Use the paths found to configure download_waymo.py")
        print("3. Run: python scripts/download_waymo.py --list-only")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("ERROR: Could not access bucket")
        print("\nTroubleshooting:")
        print("1. Check authentication: gcloud auth list")
        print("2. Ensure you have access to Waymo Open Dataset")
        print("3. Verify bucket name is correct")
        print("="*60)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
