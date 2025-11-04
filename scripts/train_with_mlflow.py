#!/usr/bin/env python3
"""Train model with automatic MLflow logging."""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.logging import print_log
from src.training.mlflow_logger import MLflowLogger


def main():
    parser = argparse.ArgumentParser(description="Train with MLflow logging")
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("--work-dir", type=str, default="work_dirs", help="Working directory")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--load-from", type=str, default=None, help="Load pretrained model")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="./mlruns", help="MLflow tracking URI")
    parser.add_argument("--mlflow-experiment", type=str, default="roadscene3d", help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    
    args = parser.parse_args()
    
    # Load config
    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    
    if args.resume_from:
        cfg.resume = True
        cfg.load_from = args.resume_from
    elif args.load_from:
        cfg.load_from = args.load_from
    
    # Setup MLflow logger
    mlflow_logger = MLflowLogger(
        experiment_name=args.mlflow_experiment,
        tracking_uri=args.mlflow_tracking_uri,
        run_name=args.run_name
    )
    
    with mlflow_logger.start_run(tags={
        "dataset": "nuscenes",
        "model": "pointpillars",
        "config": Path(args.config).stem
    }):
        # Log config
        mlflow_logger.log_config(args.config)
        
        # Log hyperparameters
        mlflow_logger.log_params({
            "model": cfg.model.type if hasattr(cfg.model, 'type') else "PointPillars",
            "batch_size": cfg.train_dataloader.batch_size,
            "num_workers": cfg.train_dataloader.num_workers,
            "learning_rate": cfg.optim_wrapper.optimizer.lr,
            "max_epochs": cfg.train_cfg.max_epochs,
            "val_interval": cfg.default_hooks.checkpoint.interval,
        })
        
        # Create runner
        runner = Runner.from_cfg(cfg)
        
        # Hook to log metrics during training
        def log_training_metrics(runner):
            """Callback to log training metrics."""
            if runner.mode == 'train':
                # Log training loss
                if hasattr(runner, 'train_loop') and hasattr(runner.train_loop, 'runner'):
                    metrics = runner.message_hub.get_scalar('train/loss')
                    if metrics:
                        mlflow_logger.log_metrics({
                            'train_loss': metrics[-1].value
                        }, step=metrics[-1].step)
            
            if runner.mode == 'val':
                # Log validation metrics from message hub
                # This is called after validation
                pass
        
        # Register hook
        runner.register_training_hooks()
        
        print_log("="*70)
        print_log("🚀 Starting Training with MLflow Logging")
        print_log("="*70)
        print_log(f"📁 Config: {args.config}")
        print_log(f"📂 Work Dir: {args.work_dir}")
        print_log(f"📊 MLflow Experiment: {args.mlflow_experiment}")
        print_log(f"🔗 Tracking URI: {args.mlflow_tracking_uri}")
        print_log("="*70)
        
        # Start training
        runner.train()
        
        # Log final checkpoint
        checkpoint_path = Path(args.work_dir) / "latest.pth"
        if checkpoint_path.exists():
            mlflow_logger.log_model(str(checkpoint_path), registered_name="roadscene3d")
        
        # Log evaluation results if available
        eval_results_path = Path(args.work_dir) / "pred_instances_3d" / "metrics_summary.json"
        if eval_results_path.exists():
            mlflow_logger.log_nuscenes_metrics(str(eval_results_path))
        
        print_log("="*70)
        print_log("✅ Training Complete! Metrics logged to MLflow")
        print_log(f"📊 View at: mlflow ui --backend-store-uri {args.mlflow_tracking_uri}")
        print_log("="*70)


if __name__ == "__main__":
    main()

