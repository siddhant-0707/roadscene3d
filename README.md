# RoadScene3D: A Self-Supervised 3D Scene Understanding & Flywheel Pipeline

A lightweight end-to-end 3D perception system that learns from RGB + LiDAR data, produces 3D bounding boxes, uses self-supervised pretraining, and includes an automated retraining flywheel with CI/CD integration.

## 🎯 Project Overview

This project implements a complete 3D object detection pipeline optimized for autonomous driving applications, with focus on:

- **3D Object Detection**: PointPillars-based detection on nuScenes dataset
- **Self-Supervised Learning**: Contrastive pretraining on LiDAR data
- **Model Optimization**: Quantization and OpenVINO export for deployment
- **Automated Flywheel**: Active learning loop with CI/CD integration
- **Production-Ready**: MLflow tracking, telemetry, and monitoring

## 🏗️ Architecture

```
Raw Sensor Data (RGB + LiDAR)
    ↓
Preprocessing & Fusion
    ↓
Self-Supervised Pretraining (Contrastive)
    ↓
3D Detection Network (PointPillars)
    ↓
Evaluation Module (mAP@IoU, Latency)
    ↓
Model Registry + Metadata Store
    ↓
Automated Retrain Trigger (New Data/Active Learning)
```

## 📋 Requirements

- **Hardware**:
  - GPU: NVIDIA RTX 4070 (8GB VRAM) or similar
  - RAM: 32GB recommended
  - Storage: ~5GB for nuScenes mini dataset (or ~400GB for full dataset)

- **Software**:
  - Python 3.9+
  - CUDA 11.8+ (for GPU training)
  - nuScenes dataset access (sign up at https://www.nuscenes.org/signup)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd roadscene3d

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install MMDetection3D (follow official installation guide)
# See: https://mmdetection3d.readthedocs.io/en/latest/get_started.html
```

### 2. Dataset Setup

```bash
# Download nuScenes mini dataset (~4GB)
# See detailed instructions in NUSCENES_DOWNLOAD.md
python scripts/download_nuscenes.py --output-dir data/nuscenes

# Or manually download from https://www.nuscenes.org/download
# Extract to data/nuscenes/v1.0-mini/
```

### 3. Training

```bash
# Train baseline model with memory-optimized settings
python src/training/train.py \
    --config configs/pointpillars_nuscenes_8gb.py \
    --work-dir work_dirs/pointpillars \
    --gpu-id 0
```

### 4. Evaluation

```python
from src.evaluation.metrics import evaluate_model
import torch

# Load model and dataloader
# ...

# Evaluate
metrics = evaluate_model(model, dataloader, device)
print(f"mAP@0.7: {metrics['mAP@0.7']:.4f}")
```

## 📁 Project Structure

```
roadscene3d/
├── data/
│   ├── nuscenes/       # nuScenes dataset (downloaded)
│   └── processed/      # Preprocessed data
├── configs/            # Model & training configs
│   ├── pointpillars_nuscenes_8gb.py  # Primary config for nuScenes
│   └── pointpillars_8gb.py            # Legacy Waymo config
├── src/
│   ├── data/           # Data loaders & preprocessing
│   │   ├── nuscenes_loader.py  # nuScenes dataset loader (primary)
│   │   └── waymo_loader.py     # Waymo loader (optional, for future use)
│   ├── training/       # Training scripts
│   │   └── train.py
│   ├── evaluation/     # Metrics & evaluation
│   │   └── metrics.py
│   ├── pretraining/    # Self-supervised pretraining
│   ├── optimization/   # Quantization & OpenVINO export
│   ├── flywheel/       # Active learning & retraining
│   └── utils/          # Utilities
│       └── memory_monitor.py
├── scripts/            # Data download & preprocessing scripts
│   └── select_waymo_subset.py
├── ci/                 # GitHub Actions workflows
├── dashboard/          # Streamlit/Gradio dashboard
├── tests/              # Unit & integration tests
├── requirements.txt
├── setup.py
└── README.md
```

## 🔧 Hardware Optimization

This project is optimized for **RTX 4070 (8GB VRAM)**:

- **Batch Size**: 2 (with gradient accumulation)
- **Mixed Precision**: FP16 training (required)
- **Gradient Accumulation**: 4 steps (effective batch size = 8)
- **Expected VRAM**: ~6-7GB peak during training
- **Expected RAM**: ~8-12GB during training

Adjust `configs/pointpillars_nuscenes_8gb.py` if you have different hardware.

## 📊 Phase Implementation

### Phase 1: Dataset & Baseline Model ✅
- [x] Environment setup
- [x] Waymo dataset integration
- [x] Baseline 3D detection model (PointPillars)
- [x] Evaluation framework

### Phase 2: Optimization & Export
- [ ] Model quantization (INT8)
- [ ] OpenVINO export
- [ ] Telemetry & logging

### Phase 3: Automated Flywheel
- [ ] Active learning pipeline
- [ ] Retraining automation
- [ ] CI/CD integration

### Phase 4: Visualization & Documentation
- [ ] Streamlit dashboard
- [ ] Self-supervised pretraining
- [ ] Documentation & article

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src
```

## 📈 Monitoring

The project uses MLflow for experiment tracking:

```bash
# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns

# View at http://localhost:5000
```

## 🤝 Contributing

This is a portfolio project. For questions or improvements, please open an issue.

## 📝 License

[Specify your license]

## 🙏 Acknowledgments

- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) for the 3D detection framework
- [nuScenes Dataset](https://www.nuscenes.org/) for the dataset
- Open source community for tools and libraries

## 📚 References

- [PointPillars Paper](https://arxiv.org/abs/1812.05784)
- [nuScenes Dataset](https://www.nuscenes.org/)
- [nuScenes DevKit](https://github.com/nutonomy/nuscenes-devkit)
- [MMDetection3D Documentation](https://mmdetection3d.readthedocs.io/)
