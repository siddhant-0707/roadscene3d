# Quick Start Guide

## Environment Setup ✅

The Python environment is ready! Verified packages:
- ✅ PyTorch 2.9.0 with CUDA 12.8
- ✅ All core dependencies installed

Activate with:
```bash
source .venv/bin/activate
```

## Download nuScenes Mini Dataset

### Step 1: Install nuScenes DevKit

```bash
source .venv/bin/activate
pip install nuscenes-devkit
```

### Step 2: Download Dataset

**Option A: Automatic (requires nuscenes-devkit)**

```bash
python scripts/download_nuscenes.py --output-dir data/nuscenes
```

**Option B: Manual Download**

1. Sign up at https://www.nuscenes.org/signup
2. Download nuScenes mini: https://www.nuscenes.org/download
3. Extract to `data/nuscenes/`

```bash
# After manual download, extract:
mkdir -p data/nuscenes
tar -xf v1.0-mini.tgz -C data/nuscenes/
```

For manual instructions:
```bash
python scripts/download_nuscenes.py --output-dir data/nuscenes --manual
```

**Storage Requirements:**
- ~3.88 GB download (Metadata and sensor file blobs)
- ~4-5 GB after extraction
- Optional: Map expansion pack (~0.37 GB)
- Files will be saved to `data/nuscenes/`

**Download Time:**
- ~2-4 minutes (much faster than Waymo!)
- Only requires internet connection (no GCP authentication needed)

### Step 3: Verify Download

After download completes, verify:

```bash
python -c "
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=True)
print(f'✅ Dataset loaded! Scenes: {len(nusc.scene)}')
"
```

## Next Steps After Download

1. **Convert to MMDetection3D format** (when ready)
2. **Install MMDetection3D** (for training)
3. **Start training baseline model**

See [README.md](README.md) for full project documentation.

## Troubleshooting

### "Permission denied" error
- Check: `gcloud auth list`
- Re-authenticate: `gcloud auth login`

### "No files found"
- Bucket structure may differ
- Run `check_waymo_bucket.py` to explore
- Check bucket name is correct

### Slow downloads
- Normal - dataset is large
- Downloads use parallel workers
- Can pause/resume (script skips existing files)
