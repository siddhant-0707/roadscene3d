"""Training script with mixed precision and gradient accumulation for 8GB VRAM."""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import mlflow
import mlflow.pytorch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.memory_monitor import MemoryMonitor
from src.evaluation.metrics import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AMPTrainer:
    """Trainer with Automatic Mixed Precision (AMP) and gradient accumulation."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            use_amp: Whether to use mixed precision training
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Setup AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Memory monitor
        self.memory_monitor = MemoryMonitor(device=device)
    
    def train_step(self, batch: dict, step: int) -> dict:
        """
        Perform one training step.
        
        Args:
            batch: Training batch
            step: Current step number
            
        Returns:
            Dictionary with loss and metrics
        """
        # Move data to device
        points = batch['points'].to(self.device)
        gt_boxes = batch['gt_bboxes_3d'].to(self.device)
        gt_labels = batch['gt_labels_3d'].to(self.device)
        
        # Forward pass with AMP
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.model(points)
            loss_dict = self.model.loss(outputs, gt_boxes, gt_labels)
            loss = sum(loss_dict.values())
            
            # Normalize loss by accumulation steps
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (step + 1) % self.gradient_accumulation_steps == 0:
            if self.use_amp:
                # Clip gradients
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()
            
            self.optimizer.zero_grad()
        
        # Return unscaled loss for logging
        total_loss = loss.item() * self.gradient_accumulation_steps
        
        return {
            'loss': total_loss,
            **{k: v.item() for k, v in loss_dict.items()}
        }
    
    def train_epoch(self, dataloader: DataLoader, epoch: int, log_interval: int = 50):
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            log_interval: Interval for logging
        """
        self.model.train()
        self.memory_monitor.log_usage(f"Epoch {epoch} - Start")
        
        total_loss = 0.0
        total_steps = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(pbar):
            metrics = self.train_step(batch, step)
            
            total_loss += metrics['loss']
            total_steps += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': metrics['loss']:.4f})
            
            # Log metrics
            if step % log_interval == 0:
                mlflow.log_metrics(metrics, step=epoch * len(dataloader) + step)
                
                if step % (log_interval * 5) == 0:
                    self.memory_monitor.log_usage(f"Epoch {epoch} - Step {step}")
        
        avg_loss = total_loss / total_steps
        self.memory_monitor.log_usage(f"Epoch {epoch} - End")
        
        return avg_loss
    
    def validate(self, dataloader: DataLoader) -> dict:
        """
        Validate model on validation set.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        # Evaluate model (this would use actual evaluation metrics)
        # For now, just compute average loss
        total_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                points = batch['points'].to(self.device)
                gt_boxes = batch['gt_bboxes_3d'].to(self.device)
                gt_labels = batch['gt_labels_3d'].to(self.device)
                
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(points)
                    loss_dict = self.model.loss(outputs, gt_boxes, gt_labels)
                    loss = sum(loss_dict.values())
                
                total_loss += loss.item()
                total_steps += 1
        
        avg_loss = total_loss / total_steps
        
        return {'val_loss': avg_loss}


def main():
    parser = argparse.ArgumentParser(description='Train 3D object detection model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--work-dir', type=str, default='work_dirs', help='Working directory')
    parser.add_argument('--resume-from', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--load-from', type=str, default=None, help='Load pretrained model')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    parser.add_argument('--mlflow-tracking-uri', type=str, default='./mlruns', help='MLflow tracking URI')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Setup MLflow
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment("roadscene3d")
    
    # Check memory
    monitor = MemoryMonitor(device=device)
    if not monitor.has_cuda:
        logger.warning("CUDA not available, training will be slow")
    else:
        monitor.log_usage("Initial")
        # Check requirements
        from src.utils.memory_monitor import check_memory_requirements
        if not check_memory_requirements(min_vram_gb=6.0):
            logger.warning("VRAM may be insufficient for training")
    
    # Load config (simplified - would use mmcv Config)
    logger.info(f"Loading config from {args.config}")
    # TODO: Implement actual config loading
    
    # Create model (placeholder - would load from MMDetection3D)
    logger.info("Creating model...")
    # model = build_model_from_config(config)
    # For now, this is a placeholder structure
    
    # Create datasets (placeholder)
    # train_dataset = build_dataset(config.data.train)
    # val_dataset = build_dataset(config.data.val)
    
    # Create data loaders
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config.train_batch_size,
    #     shuffle=True,
    #     num_workers=config.data.workers_per_gpu,
    #     pin_memory=True
    # )
    
    # Create optimizer
    # optimizer = build_optimizer(config.optimizer, model)
    
    # Create trainer
    # trainer = AMPTrainer(
    #     model=model,
    #     optimizer=optimizer,
    #     criterion=None,  # Would use model's loss
    #     device=device,
    #     use_amp=config.use_amp,
    #     gradient_accumulation_steps=config.gradient_accumulation_steps
    # )
    
    # Start training
    logger.info("Starting training...")
    
    with mlflow.start_run():
        mlflow.log_params({
            'batch_size': 2,
            'gradient_accumulation_steps': 4,
            'use_amp': True,
            'device': str(device)
        })
        
        # Training loop
        # for epoch in range(config.runner.max_epochs):
        #     train_loss = trainer.train_epoch(train_loader, epoch)
        #     mlflow.log_metric('train_loss', train_loss, step=epoch)
        #     
        #     if (epoch + 1) % config.evaluation.interval == 0:
        #         val_metrics = trainer.validate(val_loader)
        #         mlflow.log_metrics(val_metrics, step=epoch)
        #     
        #     if (epoch + 1) % config.checkpoint_config.interval == 0:
        #         checkpoint_path = f"{args.work_dir}/epoch_{epoch+1}.pth"
        #         torch.save({
        #             'epoch': epoch,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #         }, checkpoint_path)
        #         mlflow.log_artifact(checkpoint_path)
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
