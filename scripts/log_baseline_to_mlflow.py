#!/usr/bin/env python3
"""Log baseline training results to MLflow."""

import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.mlflow_logger import MLflowLogger

def main():
    # Load metrics
    metrics_file = project_root / "work_dirs/nuscenes_mini_results/pred_instances_3d/metrics_summary.json"
    
    if not metrics_file.exists():
        print(f"Error: Metrics file not found: {metrics_file}")
        return
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Initialize MLflow logger
    mlflow_logger = MLflowLogger(
        experiment_name="roadscene3d",
        tracking_uri="./mlruns",
        run_name="baseline_pointpillars_nuscenes_mini"
    )
    
    with mlflow_logger.start_run(tags={
        "dataset": "nuscenes-mini",
        "model": "pointpillars",
        "baseline": "true",
        "epochs": "40"
    }):
        # Log config
        config_path = project_root / "configs/pointpillars_nuscenes_mini.py"
        if config_path.exists():
            mlflow_logger.log_config(str(config_path))
        
        # Log hyperparameters
        mlflow_logger.log_params({
            "model": "PointPillars",
            "dataset": "nuScenes v1.0-mini",
            "batch_size": 1,
            "num_workers": 2,
            "max_epochs": 40,
            "learning_rate": 0.001,
        })
        
        # Log nuScenes metrics
        mlflow_logger.log_nuscenes_metrics(str(metrics_file))
        
        # Log checkpoint
        checkpoint_path = project_root / "work_dirs/pointpillars_nuscenes_baseline/epoch_40.pth"
        if checkpoint_path.exists():
            mlflow_logger.log_model(str(checkpoint_path), registered_name="roadscene3d")
        else:
            print(f"⚠️  Checkpoint not found at: {checkpoint_path}")
        
        print("="*70)
        print("✅ Baseline training results logged to MLflow!")
        print(f"📊 mAP: {metrics.get('mean_ap', 0)*100:.2f}%")
        print(f"📊 NDS: {metrics.get('nd_score', 0)*100:.2f}%")
        print("="*70)
        print("\n🚀 View results:")
        print("   • Streamlit Dashboard: streamlit run dashboard/app.py")
        print("   • MLflow UI: mlflow ui --backend-store-uri ./mlruns")
        print("="*70)

if __name__ == "__main__":
    main()

