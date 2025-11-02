#!/bin/bash
# Installation script for MMDetection3D

set -e  # Exit on error

echo "=========================================="
echo "MMDetection3D Installation Script"
echo "=========================================="
echo ""

# Check if in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "   Activate with: source .venv/bin/activate"
    exit 1
fi

echo "✅ Virtual environment: $VIRTUAL_ENV"
echo ""

# Check PyTorch and CUDA
echo "Checking PyTorch and CUDA..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
" || exit 1

echo ""
echo "=========================================="
echo "Step 1: Installing mmcv-full"
echo "=========================================="

# Determine CUDA version for wheel
# PyTorch 2.9 with CUDA 12.8 -> try cu121 or cu118 wheels
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])" | cut -d. -f1,2)
CUDA_VERSION="cu121"  # CUDA 12.1 wheels work with 12.8

echo "Attempting to install mmcv-full for PyTorch ${TORCH_VERSION} and CUDA ${CUDA_VERSION}..."
echo ""

# Try PyTorch 2.9 wheel first, then fall back to 2.1 or 2.0
for TORCH_VER in "2.9" "2.1" "2.0" "1.13"; do
    WHEEL_URL="https://download.openmmlab.com/mmcv/dist/${CUDA_VERSION}/torch${TORCH_VER}.0/index.html"
    echo ""
    echo "Trying PyTorch ${TORCH_VER}.0 wheel..."
    echo "URL: ${WHEEL_URL}"
    
    if pip install mmcv-full -f ${WHEEL_URL} 2>&1 | tee /tmp/mmcv_install.log; then
        echo "✅ Successfully installed mmcv-full for PyTorch ${TORCH_VER}.0"
        break
    else
        echo "❌ Failed for PyTorch ${TORCH_VER}.0, trying next version..."
        if [ "$TORCH_VER" = "1.13" ]; then
            echo ""
            echo "⚠️  All wheel versions failed. Trying to build from source..."
            echo "This may take 15-30 minutes."
            pip install mmcv-full || {
                echo "❌ Failed to install mmcv-full"
                exit 1
            }
            break
        fi
    fi
done

echo ""
echo "Verifying mmcv-full installation..."
python -c "import mmcv; print(f'✅ mmcv version: {mmcv.__version__}')" || {
    echo "❌ mmcv-full verification failed"
    exit 1
}

echo ""
echo "=========================================="
echo "Step 2: Installing MMDetection"
echo "=========================================="

echo "Installing mmdet..."
pip install git+https://github.com/open-mmlab/mmdetection.git || {
    echo "❌ Failed to install mmdetection"
    exit 1
}

echo ""
echo "Verifying mmdet installation..."
python -c "import mmdet; print(f'✅ mmdet version: {mmdet.__version__}')" || {
    echo "❌ mmdet verification failed"
    exit 1
}

echo ""
echo "=========================================="
echo "Step 3: Installing MMDetection3D"
echo "=========================================="

# Clone mmdetection3d if not already present
if [ ! -d "mmdetection3d" ]; then
    echo "Cloning MMDetection3D repository..."
    git clone https://github.com/open-mmlab/mmdetection3d.git
fi

cd mmdetection3d

echo "Installing MMDetection3D..."
pip install -v -e . || {
    echo "❌ Failed to install mmdetection3d"
    exit 1
}

cd ..

echo ""
echo "=========================================="
echo "Step 4: Verifying Installation"
echo "=========================================="

python -c "
import mmcv
import mmdet
import mmdet3d
print('✅ mmcv version:', mmcv.__version__)
print('✅ mmdet version:', mmdet.__version__)
print('✅ mmdet3d version:', mmdet3d.__version__)
print('')
print('==========================================')
print('✅ All packages installed successfully!')
print('==========================================')
" || {
    echo "❌ Verification failed"
    exit 1
}

echo ""
echo "Installation complete! You can now use MMDetection3D."
echo ""
echo "Next steps:"
echo "1. Verify with: python -c 'import mmdet3d; print(mmdet3d.__version__)'"
echo "2. Start training: python src/training/train.py --config configs/pointpillars_nuscenes_8gb.py"
