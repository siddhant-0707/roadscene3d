# Environment Setup Guide

## ✅ Completed Setup

The Python environment has been created using `uv` with Python 3.12.

**Status**: ✅ All core packages verified and working!

### Installed Packages

**Core ML Frameworks:**
- ✅ torch, torchvision (with CUDA 12.8 support)

**Data Processing:**
- ✅ open3d, numpy, scipy, pandas, h5py
- ✅ tqdm, pyyaml, python-dotenv, psutil

**MLOps & Monitoring:**
- ✅ mlflow, opentelemetry-api, opentelemetry-sdk
- ✅ streamlit, plotly, tensorboard

**Workflow:**
- ✅ prefect, pytest, pytest-benchmark

**Other:**
- ✅ tensorflow

## ⚠️ Packages Requiring Special Installation

### 1. MMDetection3D Ecosystem ✅ INSTALLED

**Your System:**
- PyTorch: 2.9.0
- CUDA: 12.8
- Python: 3.12

**Installed Versions:**
- mmcv: 2.1.0 (for PyTorch 2.4 - closest available to 2.9)
- mmengine: 0.10.7
- mmdet: 3.3.0
- mmdet3d: 1.4.0

**Installation completed!** All packages are working correctly.

**Note**: 
- For PyTorch 2.9, we used mmcv wheels built for PyTorch 2.4 (latest available)
- Newer MMDetection uses `mmcv` (v2.x) instead of `mmcv-full` (v1.x)
- MMDetection3D is installed in editable mode from cloned repositories

### 2. Waymo Open Dataset Tools

The `waymo-open-dataset-tf` package is not available via PyPI. Install from source:

```bash
source .venv/bin/activate

# Clone and install waymo-open-dataset
git clone https://github.com/waymo-research/waymo-open-dataset.git
cd waymo-open-dataset
pip install .
cd ..
```

Or download the pre-built wheel from Waymo's releases if available.

Alternatively, you can use the dataset conversion tools from MMDetection3D which may have Waymo support built-in.

## Activating the Environment

```bash
source .venv/bin/activate
```

Or with `uv`:

```bash
uv run python script.py
```

## Verifying Installation

```bash
source .venv/bin/activate

# Check PyTorch
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Check other packages
python -c "import open3d, numpy, pandas, mlflow, streamlit; print('Core packages OK')"
```

## Next Steps

### 1. Download Waymo Dataset (Ready Now!)

Your environment is ready to download the dataset:

```bash
# First, check what's available in the bucket
python scripts/check_waymo_bucket.py --bucket gs://waymo_open_dataset_v_1_4_3

# Or list files without downloading
python scripts/download_waymo.py \
    --bucket gs://waymo_open_dataset_v_1_4_3 \
    --list-only

# Download a subset (120 train, 25 val, 25 test)
python scripts/download_waymo.py \
    --output-dir data/waymo \
    --n-train 120 \
    --n-val 25 \
    --n-test 25 \
    --bucket gs://waymo_open_dataset_v_1_4_3
```

See [README_DOWNLOAD.md](README_DOWNLOAD.md) for detailed download instructions.

### 2. Install MMDetection3D (When Ready for Training)

You can proceed with dataset download first. MMDetection3D installation can wait until you're ready to train:

```bash
source .venv/bin/activate

# Install mmcv-full (may need to check for torch2.9 wheels)
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.9/index.html

# If torch2.9 wheels don't exist, may need torch 2.0/2.1 wheels or build from source
pip install mmdet mmdet3d
```

### 3. Start Development

- ✅ Environment is ready for dataset download scripts
- ✅ All core dependencies installed and verified
- ⏳ MMDetection3D can be added when needed for training

## Notes

- Python 3.12 was used (Python 3.14 was too new for many packages)
- All packages installed via `uv` for speed
- Virtual environment location: `.venv/`
- For production training, install mmcv-full and mmdet3d with proper CUDA wheels
