"""MLflow model registry for versioning and promotion."""

import mlflow
import mlflow.pytorch
from pathlib import Path
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manage model versions and promotions using MLflow."""
    
    def __init__(self, tracking_uri: str = "./mlruns", registry_name: str = "roadscene3d"):
        """
        Initialize model registry.
        
        Args:
            tracking_uri: MLflow tracking URI
            registry_name: Name of the model registry
        """
        mlflow.set_tracking_uri(tracking_uri)
        self.registry_name = registry_name
    
    def register_model(
        self,
        model_path: str,
        metrics: Dict[str, float],
        metadata: Dict,
        stage: str = "None"
    ) -> str:
        """
        Register a new model version.
        
        Args:
            model_path: Path to model checkpoint
            metrics: Dictionary of evaluation metrics
            metadata: Additional metadata (config, dataset info, etc.)
            stage: Initial stage (None, Staging, Production)
            
        Returns:
            Model version URI
        """
        with mlflow.start_run():
            # Log model
            mlflow.pytorch.log_model(
                pytorch_model=model_path,
                artifact_path="model",
                registered_model_name=self.registry_name
            )
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log metadata
            mlflow.log_params(metadata)
            
            run_id = mlflow.active_run().info.run_id
        
        # Promote to stage if specified
        if stage != "None":
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=self.registry_name,
                version=self.get_latest_version(),
                stage=stage
            )
        
        logger.info(f"Registered model version {run_id} at stage {stage}")
        return run_id
    
    def get_latest_version(self, stage: Optional[str] = None) -> int:
        """
        Get latest model version.
        
        Args:
            stage: Optional stage filter
            
        Returns:
            Latest version number
        """
        client = mlflow.tracking.MlflowClient()
        if stage:
            versions = client.get_latest_versions(self.registry_name, stages=[stage])
        else:
            versions = client.search_model_versions(f"name='{self.registry_name}'")
        
        if not versions:
            return 0
        
        return max([v.version for v in versions])
    
    def promote_model(self, version: int, stage: str):
        """
        Promote model to a new stage.
        
        Args:
            version: Model version number
            stage: Target stage (Staging, Production)
        """
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=self.registry_name,
            version=version,
            stage=stage
        )
        logger.info(f"Promoted model version {version} to {stage}")
    
    def load_model(self, stage: str = "Production"):
        """
        Load model from registry.
        
        Args:
            stage: Stage to load from
            
        Returns:
            Loaded PyTorch model
        """
        model_uri = f"models:/{self.registry_name}/{stage}"
        model = mlflow.pytorch.load_model(model_uri)
        logger.info(f"Loaded model from {model_uri}")
        return model
    
    def compare_models(self, version1: int, version2: int) -> Dict:
        """
        Compare two model versions.
        
        Args:
            version1: First model version
            version2: Second model version
            
        Returns:
            Comparison dictionary
        """
        client = mlflow.tracking.MlflowClient()
        
        v1_info = client.get_model_version(self.registry_name, version1)
        v2_info = client.get_model_version(self.registry_name, version2)
        
        # Get run metrics
        run1 = client.get_run(v1_info.run_id)
        run2 = client.get_run(v2_info.run_id)
        
        comparison = {
            'version1': {
                'version': version1,
                'stage': v1_info.current_stage,
                'metrics': run1.data.metrics
            },
            'version2': {
                'version': version2,
                'stage': v2_info.current_stage,
                'metrics': run2.data.metrics
            }
        }
        
        return comparison
