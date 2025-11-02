#!/bin/bash
# Training script with memory optimizations for 8GB GPU

cd /home/sidd/Documents/GitHub/roadscene3d
source .venv/bin/activate

# Set PyTorch memory allocation to avoid fragmentation and reduce memory caching
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Start training (no --amp flag due to norm layer incompatibility with float16)
python mmdetection3d/tools/train.py \
    configs/pointpillars_nuscenes_mini.py \
    --work-dir work_dirs/pointpillars_nuscenes_baseline

