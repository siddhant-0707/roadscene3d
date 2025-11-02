# Downloading Waymo Open Dataset

This guide explains how to download the Waymo Open Dataset using the provided scripts.

## Prerequisites

1. **Google Cloud SDK installed**: You should have `gcloud` CLI installed
2. **Authenticated**: You must be authenticated with the Google account that has Waymo dataset access
3. **Dataset Access**: Ensure you've signed up for Waymo Open Dataset access

## Step 1: Check Authentication

First, verify you're authenticated:

```bash
gcloud auth list
```

If not authenticated, run:

```bash
gcloud auth login
```

## Step 2: Check Bucket Structure

The bucket structure may vary. Use the checker script to explore:

```bash
python scripts/check_waymo_bucket.py --bucket gs://waymo_open_dataset_v_1_4_3
```

Or manually check:

```bash
gcloud storage ls gs://waymo_open_dataset_v_1_4_3/
```

Common buckets:
- `gs://waymo_open_dataset_v_1_4_3` - Perception v1.4.3 (with maps)
- `gs://waymo_open_dataset_v_2_0_1_individual_files` - Perception v2.0.1 (modular format)

## Step 3: List Available Files

Before downloading, see what's available:

```bash
python scripts/download_waymo.py \
    --bucket gs://waymo_open_dataset_v_1_4_3 \
    --list-only
```

This will show:
- Bucket structure
- Number of .tfrecord files
- Sample file paths

## Step 4: Download Dataset Subset

### For v1.4.3 (recommended for initial setup)

Download training, validation, and test subsets:

```bash
python scripts/download_waymo.py \
    --output-dir data/waymo \
    --n-train 120 \
    --n-val 25 \
    --n-test 25 \
    --bucket gs://waymo_open_dataset_v_1_4_3
```

**Note**: For v1.4.3, each file contains all components (LiDAR, camera, labels), so the `--components` flag is ignored.

### For v2.0.1 (modular format)

Download specific components:

```bash
python scripts/download_waymo.py \
    --output-dir data/waymo \
    --components lidar 3d_labels \
    --n-train 120 \
    --n-val 25 \
    --n-test 25 \
    --bucket gs://waymo_open_dataset_v_2_0_1_individual_files
```

Available components:
- `camera` - Camera images
- `lidar` - LiDAR point clouds
- `labels` - 2D labels
- `3d_labels` - 3D bounding box labels

## Download Options

```bash
python scripts/download_waymo.py --help
```

Key options:
- `--output-dir`: Where to save files (default: `data/waymo`)
- `--n-train`: Number of training files (default: 120)
- `--n-val`: Number of validation files (default: 25)
- `--n-test`: Number of test files (default: 25)
- `--bucket`: GCS bucket path
- `--list-only`: Only list files, don't download
- `--components`: Components to download (v2.0.1 only)

## Download Progress

The script will:
1. Check authentication
2. List available files
3. Filter to requested subset
4. Download files in parallel (4 workers)
5. Save download manifest to `data/waymo/download_manifest.json`

## Storage Requirements

Approximate storage for subsets:
- **120 training files**: ~50-100 GB
- **25 validation files**: ~10-20 GB
- **25 test files**: ~10-20 GB
- **Total subset**: ~70-140 GB

Full dataset is much larger (~500+ GB).

## Troubleshooting

### "Permission denied" or "Access denied"

- Ensure you're authenticated: `gcloud auth list`
- Verify you have Waymo dataset access
- Check bucket name is correct

### "No files found"

- Bucket structure may differ
- Run `--list-only` to explore bucket structure
- Try different bucket paths
- Check with: `gcloud storage ls gs://waymo_open_dataset_v_*/`

### Slow downloads

- Downloads use parallel workers (4 by default)
- Large files may take time
- Ensure stable internet connection
- Consider downloading smaller subset first

### Resume interrupted downloads

The script will skip files that already exist. To re-download:

```bash
rm -rf data/waymo/training/*  # Remove specific split
python scripts/download_waymo.py ...  # Re-run download
```

## Manual Download (Alternative)

If scripts don't work, you can download manually:

```bash
# Download specific file
gcloud storage cp \
    gs://waymo_open_dataset_v_1_4_3/perception/training/segment-*.tfrecord \
    data/waymo/training/

# Download entire directory (large!)
gcloud storage cp -r \
    gs://waymo_open_dataset_v_1_4_3/perception/training/ \
    data/waymo/training/
```

## Next Steps

After downloading:

1. **Verify downloads**: Check that files exist in `data/waymo/`
2. **Check manifest**: Review `data/waymo/download_manifest.json`
3. **Convert to MMDetection3D format**: Use the data loader (see README.md)
4. **Start training**: Run training script

For more information, see the main [README.md](README.md).
