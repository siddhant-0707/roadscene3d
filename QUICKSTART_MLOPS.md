# 🚀 Quick Start - MLOps Dashboard

## View Your Results in 2 Minutes

### Option 1: Streamlit Dashboard (Recommended) ⭐

```bash
streamlit run dashboard/app.py
```

**Open:** http://localhost:8501

**What you'll see:**
- 📊 Current model performance (mAP: 4.25%, NDS: 10.44%)
- 📈 Performance trends over training runs
- 🎯 Per-class accuracy (cars, pedestrians, etc.)
- ⚡ Speed-accuracy trade-off analysis

---

### Option 2: MLflow UI (Advanced)

```bash
bash scripts/start_mlflow_server.sh
# Or: mlflow ui --backend-store-uri ./mlruns
```

**Open:** http://localhost:5000

**What you'll see:**
- Detailed experiment runs
- Model checkpoints and artifacts
- Parameter history
- Model registry

---

## What's Already Logged?

Your baseline training results are already in MLflow:

✅ **Metrics**: mAP, NDS, error metrics  
✅ **Per-Class AP**: Car (25.18%), Pedestrian (17.31%), etc.  
✅ **Checkpoint**: Model saved at epoch 40  
✅ **Config**: Training configuration file  

---

## Next: Train More Models

When you train new models, use:

```bash
python scripts/train_with_mlflow.py \
    configs/pointpillars_nuscenes_mini.py \
    --work-dir work_dirs/my_experiment \
    --run-name my_experiment_v1
```

All metrics will automatically appear in both dashboards! 🎉

---

**Need help?** See [docs/MLOPS_SETUP.md](docs/MLOPS_SETUP.md)

