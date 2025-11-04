"""MLflow integration for training and evaluation logging."""

import mlflow
import mlflow.pytorch
from pathlib import Path
from typing import Dict, Optional, Any
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MLflowLogger:
    """Automatic MLflow logging for MMDetection3D training."""
    
    def __init__(
        self,
        experiment_name: str = "roadscene3d",
        tracking_uri: str = "./mlruns",
        run_name: Optional[str] = None
    ):
        """
        Initialize MLflow logger.
        
        Args:
            experiment_name: Name of MLflow experiment
            tracking_uri: MLflow tracking URI
            run_name: Optional run name (defaults to timestamp)
        """
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        self.run_name = run_name or f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_run = None
    
    def start_run(self, tags: Optional[Dict[str, str]] = None):
        """Start a new MLflow run."""
        self.active_run = mlflow.start_run(run_name=self.run_name, tags=tags or {})
        logger.info(f"Started MLflow run: {self.active_run.info.run_id}")
        return self.active_run
    
    def end_run(self):
        """End the current MLflow run."""
        if self.active_run:
            mlflow.end_run()
            logger.info("Ended MLflow run")
    
    def log_config(self, config_path: str):
        """
        Log training configuration file.
        
        Args:
            config_path: Path to config file
        """
        config_path = Path(config_path)
        if config_path.exists():
            mlflow.log_artifact(str(config_path), artifact_path="configs")
            logger.info(f"Logged config: {config_path}")
        else:
            logger.warning(f"Config file not found: {config_path}")
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log hyperparameters.
        
        Args:
            params: Dictionary of parameters
        """
        # Convert nested dicts to dot notation for MLflow
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)
        logger.debug(f"Logged {len(flat_params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged {len(metrics)} metrics at step {step}")
    
    def log_model(self, checkpoint_path: str, registered_name: Optional[str] = None):
        """
        Log model checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            registered_name: Optional registered model name for model registry
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        
        # Log checkpoint as artifact
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
        logger.info(f"Logged checkpoint: {checkpoint_path}")
        
        # For model registry, we need to log the checkpoint path as metadata
        # since MMDetection3D models need special loading
        if registered_name:
            # Log checkpoint metadata for model registry
            mlflow.log_params({
                "checkpoint_path": str(checkpoint_path),
                "model_type": "mmdet3d_checkpoint"
            })
            
            # Note: For full model registration, we'd need to load the model
            # and save it in MLflow format, but MMDetection3D checkpoints
            # require special handling. For now, we log the checkpoint path.
            logger.info(f"Checkpoint logged for registry: {registered_name}")
            logger.info("Note: To fully register, load model with MMDetection3D and use mlflow.pytorch.log_model()")
    
    def log_nuscenes_metrics(self, metrics_file: str):
        """
        Log nuScenes evaluation metrics from results file.
        
        Args:
            metrics_file: Path to metrics_summary.json
        """
        metrics_path = Path(metrics_file)
        if not metrics_path.exists():
            logger.warning(f"Metrics file not found: {metrics_path}")
            return
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Extract key metrics
        mlflow_metrics = {
            'mAP': metrics.get('mean_ap', 0.0),
            'NDS': metrics.get('nd_score', 0.0),
            'mATE': metrics.get('tp_errors', {}).get('trans_err', 0.0),
            'mASE': metrics.get('tp_errors', {}).get('scale_err', 0.0),
            'mAOE': metrics.get('tp_errors', {}).get('orient_err', 0.0),
            'mAVE': metrics.get('tp_errors', {}).get('vel_err', 0.0),
            'mAAE': metrics.get('tp_errors', {}).get('attr_err', 0.0),
        }
        
        # Log per-class AP
        for class_name, ap in metrics.get('mean_dist_aps', {}).items():
            mlflow_metrics[f'AP_{class_name}'] = ap
        
        self.log_metrics(mlflow_metrics)
        
        # Log full metrics file as artifact
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
        
        logger.info(f"Logged nuScenes metrics: mAP={mlflow_metrics['mAP']:.4f}, NDS={mlflow_metrics['NDS']:.4f}")
    
    def log_inference_speed(self, latency_ms: float, throughput_fps: float):
        """
        Log inference speed metrics.
        
        Args:
            latency_ms: Average latency in milliseconds
            throughput_fps: Throughput in frames per second
        """
        self.log_metrics({
            'latency_mean_ms': latency_ms,
            'throughput_fps': throughput_fps,
        })
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary for MLflow."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, str(v) if not isinstance(v, (int, float, bool, str)) else v))
        return dict(items)
    
    def __enter__(self):
        """Context manager entry."""
        self.start_run()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()

