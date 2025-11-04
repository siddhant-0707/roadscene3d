# MLOps & Flywheel Pipeline Setup

## 🎉 What We've Built

Your RoadScene3D project now has a complete **MLOps infrastructure** with:

### ✅ Experiment Tracking (MLflow)
- **Automatic metric logging** during training
- **Model versioning** and registry
- **Experiment comparison** and history
- **Baseline results logged** (mAP: 4.25%, NDS: 10.44%)

### ✅ Interactive Dashboards
- **Streamlit dashboard** for stakeholders
- **MLflow UI** for detailed experiment tracking
- **Real-time metrics** visualization

### ✅ Automation Pipeline
- **Active learning** with uncertainty sampling
- **Automated retraining** workflow (Prefect)
- **Quality gates** for model promotion

### ✅ 3D Visualization
- **Point cloud visualization** with predictions
- **Ground truth comparison**
- **Interactive 3D viewer**

---

## 🚀 Quick Start Guide

### 1. View MLflow Results

**Option A: Streamlit Dashboard (Recommended for non-technical users)**
```bash
streamlit run dashboard/app.py
```
Then open: http://localhost:8501

**Features:**
- 📊 Overview metrics (mAP, NDS, latency)
- 📈 Performance trends over time
- 🎯 Per-class accuracy breakdown
- ⚡ Speed-accuracy trade-off analysis

**Option B: MLflow UI (For technical users)**
```bash
bash scripts/start_mlflow_server.sh
# Or manually:
mlflow ui --backend-store-uri ./mlruns
```
Then open: http://localhost:5000

**Features:**
- Detailed experiment runs
- Model artifacts and checkpoints
- Parameter and metric history
- Model registry

---

### 2. Train with MLflow Logging

For future training runs, use the MLflow-integrated training script:

```bash
python scripts/train_with_mlflow.py \
    configs/pointpillars_nuscenes_mini.py \
    --work-dir work_dirs/my_experiment \
    --mlflow-experiment roadscene3d \
    --run-name my_experiment_v1
```

This will automatically:
- ✅ Log all hyperparameters
- ✅ Track training metrics
- ✅ Save model checkpoints
- ✅ Log evaluation results

---

### 3. Visualize Predictions

View 3D point clouds with predictions:

```bash
python scripts/visualize_predictions.py \
    configs/pointpillars_nuscenes_mini.py \
    work_dirs/pointpillars_nuscenes_baseline/epoch_40.pth \
    --sample-idx 0 \
    --show
```

---

### 4. Run Active Learning

Select uncertain samples for labeling:

```python
from src.flywheel.active_learning import UncertaintySampler, ActiveLearningPipeline

# Initialize sampler
sampler = UncertaintySampler(strategy='entropy', top_k=100)

# Run active learning iteration
pipeline = ActiveLearningPipeline(
    model=model,
    labeled_data=train_set,
    unlabeled_data=unlabeled_pool,
    uncertainty_sampler=sampler,
    device=device
)

results = pipeline.run_iteration()
```

---

### 5. Automated Retraining

Set up automated retraining with Prefect:

```python
from src.flywheel.prefect_workflow import automated_retraining_flow

# Run workflow
result = automated_retraining_flow(
    data_path="data/unlabeled",
    tracking_uri="./mlruns",
    work_dir="work_dirs/retraining"
)
```

Or schedule it with Prefect Cloud/Server.

---

## 📊 Current Baseline Results

Your baseline model is now tracked in MLflow:

| Metric | Value |
|--------|-------|
| **mAP** | **4.25%** |
| **NDS** | **10.44%** |
| **Car AP** | 25.18% |
| **Pedestrian AP** | 17.31% |

**View in:**
- Streamlit: `streamlit run dashboard/app.py`
- MLflow UI: `mlflow ui --backend-store-uri ./mlruns`

---

## 🎯 Next Steps

### For Non-Technical Stakeholders:
1. **View Dashboard**: `streamlit run dashboard/app.py`
2. **Check Progress**: See mAP progression and per-class performance
3. **Monitor Quality**: Track error metrics (translation, rotation, etc.)

### For ML Engineers:
1. **Compare Experiments**: Use MLflow UI to compare different runs
2. **Tune Hyperparameters**: Track parameter changes and their impact
3. **Monitor Training**: Check training loss curves and validation metrics

### For DevOps:
1. **Set Up Prefect**: Deploy automated retraining workflow
2. **CI/CD Integration**: Add evaluation gates to GitHub Actions
3. **Model Registry**: Promote models from Staging → Production

---

## 📁 Project Structure

```
roadscene3d/
├── mlruns/                    # MLflow tracking data
│   └── 0/                     # Experiment: roadscene3d
│       └── [run_ids]/         # Individual training runs
│
├── dashboard/
│   └── app.py                 # Streamlit dashboard
│
├── src/
│   ├── training/
│   │   └── mlflow_logger.py   # MLflow integration
│   ├── flywheel/
│   │   ├── active_learning.py  # Active learning pipeline
│   │   └── prefect_workflow.py # Automated retraining
│   ├── visualization/
│   │   └── pointcloud_viz.py  # 3D visualization
│   └── models/
│       └── registry.py        # Model registry
│
└── scripts/
    ├── train_with_mlflow.py    # Training with MLflow
    ├── log_baseline_to_mlflow.py
    ├── visualize_predictions.py
    └── start_mlflow_server.sh
```

---

## 🎨 Dashboard Features

### Overview Metrics
- Latest mAP and NDS
- Best performance so far
- Inference latency
- Total training runs

### Performance Charts
1. **Performance Trends**: mAP and NDS over time
2. **Error Metrics**: Translation, rotation, scale errors
3. **Per-Class AP**: Accuracy breakdown by object class
4. **Speed-Accuracy**: Trade-off analysis

### Model Registry
- Production and Staging versions
- Model comparison
- Version history

---

## 🔧 Troubleshooting

### MLflow not tracking?
- Check `./mlruns` directory exists
- Verify MLflow is installed: `pip list | grep mlflow`
- Check experiment name matches in code

### Dashboard shows no data?
- Run `python scripts/log_baseline_to_mlflow.py` first
- Check MLflow tracking URI matches in dashboard
- Refresh data in dashboard sidebar

### Visualization not working?
- Ensure Open3D is installed: `pip install open3d`
- Check checkpoint path is correct
- Verify sample index is within dataset range

---

## 📚 Additional Resources

- **MLflow Docs**: https://mlflow.org/docs/latest/index.html
- **Streamlit Docs**: https://docs.streamlit.io/
- **Prefect Docs**: https://docs.prefect.io/
- **Training Results**: [docs/TRAINING_RESULTS.md](TRAINING_RESULTS.md)

---

## 🎉 Summary

You now have a **production-ready MLOps pipeline** with:

✅ **Experiment Tracking** - Every training run is logged  
✅ **Interactive Dashboards** - Beautiful visualizations for stakeholders  
✅ **Automated Workflows** - Active learning and retraining  
✅ **Model Registry** - Version control for models  
✅ **3D Visualization** - See predictions in action  

**Your project is ready to impress both technical and non-technical audiences!** 🚀

