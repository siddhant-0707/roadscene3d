"""Script to select a balanced subset of Waymo Open Dataset segments.

This script helps create a manageable dataset subset for training while maintaining
scene diversity (highway, urban, residential).
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict
import random

# Note: This is a placeholder script structure
# Actual implementation requires Waymo Open Dataset tools and segment metadata


def parse_segment_info(segment_path: str) -> Dict:
    """
    Parse segment info to determine scene type.
    
    Args:
        segment_path: Path to Waymo segment file
        
    Returns:
        Dictionary with segment metadata (type, location, etc.)
    """
    # TODO: Implement actual parsing using waymo_open_dataset library
    # This would require loading the TFRecord file and extracting metadata
    return {
        'path': segment_path,
        'type': 'unknown',  # highway, urban, residential
        'segment_id': os.path.basename(segment_path)
    }


def select_balanced_subset(
    all_segments: List[str],
    n_train: int = 120,
    n_val: int = 25,
    n_test: int = 25,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Select a balanced subset of segments for train/val/test splits.
    
    Args:
        all_segments: List of paths to all segment files
        n_train: Number of training segments
        n_val: Number of validation segments
        n_test: Number of test segments
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing segment paths
    """
    random.seed(seed)
    
    # Parse segment info (in real implementation, this would categorize by scene type)
    segment_info = [parse_segment_info(seg) for seg in all_segments]
    
    # For now, use random sampling
    # In production, you'd want to balance by scene type, weather, time of day, etc.
    shuffled = segment_info.copy()
    random.shuffle(shuffled)
    
    # Split into train/val/test
    train_segments = shuffled[:n_train]
    val_segments = shuffled[n_train:n_train + n_val]
    test_segments = shuffled[n_train + n_val:n_train + n_val + n_test]
    
    return {
        'train': [s['path'] for s in train_segments],
        'val': [s['path'] for s in val_segments],
        'test': [s['path'] for s in test_segments]
    }


def save_subset_splits(splits: Dict[str, List[str]], output_dir: Path):
    """
    Save subset splits to JSON files.
    
    Args:
        splits: Dictionary with train/val/test segment lists
        output_dir: Directory to save split files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, segments in splits.items():
        output_path = output_dir / f"{split_name}_segments.json"
        with open(output_path, 'w') as f:
            json.dump(segments, f, indent=2)
        print(f"Saved {split_name} split with {len(segments)} segments to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Select balanced subset of Waymo Open Dataset"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing Waymo segment files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for split JSON files'
    )
    parser.add_argument(
        '--n-train',
        type=int,
        default=120,
        help='Number of training segments'
    )
    parser.add_argument(
        '--n-val',
        type=int,
        default=25,
        help='Number of validation segments'
    )
    parser.add_argument(
        '--n-test',
        type=int,
        default=25,
        help='Number of test segments'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Find all segment files (TFRecord format)
    data_dir = Path(args.data_dir)
    segment_files = list(data_dir.glob('*.tfrecord*'))
    
    if not segment_files:
        print(f"Warning: No segment files found in {data_dir}")
        print("Expected .tfrecord or .tfrecord.gz files")
        return
    
    print(f"Found {len(segment_files)} segment files")
    
    # Select balanced subset
    splits = select_balanced_subset(
        [str(f) for f in segment_files],
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed
    )
    
    # Save splits
    output_dir = Path(args.output_dir)
    save_subset_splits(splits, output_dir)
    
    print(f"\nSubset selection complete:")
    print(f"  Training: {len(splits['train'])} segments")
    print(f"  Validation: {len(splits['val'])} segments")
    print(f"  Test: {len(splits['test'])} segments")


if __name__ == '__main__':
    main()
