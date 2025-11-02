"""Active learning pipeline with uncertainty sampling."""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_prediction_entropy(predictions: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of predictions for uncertainty estimation.
    
    Args:
        predictions: Model predictions [B, N, C] or [B, N]
        
    Returns:
        Entropy values [B, N]
    """
    if len(predictions.shape) == 3:
        # Softmax to get probabilities
        probs = torch.softmax(predictions, dim=-1)
        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    else:
        # For binary or regression, compute variance-based uncertainty
        entropy = torch.var(predictions, dim=-1) if len(predictions.shape) > 2 else predictions
    
    return entropy


def compute_prediction_margin(predictions: torch.Tensor) -> torch.Tensor:
    """
    Compute margin (difference between top-2 predictions) for uncertainty.
    
    Args:
        predictions: Model predictions [B, N, C]
        
    Returns:
        Margin values [B, N]
    """
    if len(predictions.shape) == 3:
        # Get top-2 predictions
        top2, _ = torch.topk(predictions, k=2, dim=-1)
        margin = top2[:, :, 0] - top2[:, :, 1]
    else:
        # For binary/regression, use standard deviation
        margin = torch.std(predictions, dim=-1) if len(predictions.shape) > 2 else predictions
    
    return margin


class UncertaintySampler:
    """Sample uncertain examples for active learning."""
    
    def __init__(
        self,
        strategy: str = 'entropy',
        top_k: int = 100
    ):
        """
        Initialize uncertainty sampler.
        
        Args:
            strategy: Sampling strategy ('entropy', 'margin', 'random')
            top_k: Number of samples to select
        """
        self.strategy = strategy
        self.top_k = top_k
        
        if strategy not in ['entropy', 'margin', 'random']:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def select_samples(
        self,
        model: nn.Module,
        unlabeled_data: List[Dict],
        device: torch.device
    ) -> List[int]:
        """
        Select uncertain samples from unlabeled pool.
        
        Args:
            model: Trained model for uncertainty estimation
            unlabeled_data: List of unlabeled data samples
            device: Device to run inference on
            
        Returns:
            List of selected sample indices
        """
        model.eval()
        
        uncertainties = []
        
        with torch.no_grad():
            for idx, sample in enumerate(unlabeled_data):
                # Get predictions
                points = sample['points'].to(device)
                
                # Forward pass
                outputs = model(points)
                
                # Extract predictions (format depends on model)
                # TODO: Adapt based on actual model output
                predictions = outputs.get('cls_scores', outputs.get('bbox_preds'))
                
                if predictions is None:
                    logger.warning(f"Could not extract predictions from model output")
                    continue
                
                # Compute uncertainty
                if self.strategy == 'entropy':
                    uncertainty = compute_prediction_entropy(predictions).mean().item()
                elif self.strategy == 'margin':
                    uncertainty = -compute_prediction_margin(predictions).mean().item()
                else:  # random
                    uncertainty = np.random.random()
                
                uncertainties.append((idx, uncertainty))
        
        # Sort by uncertainty (highest first)
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        
        # Select top-k
        selected_indices = [idx for idx, _ in uncertainties[:self.top_k]]
        
        logger.info(
            f"Selected {len(selected_indices)} samples using {self.strategy} strategy"
        )
        
        return selected_indices


class PseudoLabeler:
    """Generate pseudo-labels for unlabeled data."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize pseudo-labeler.
        
        Args:
            confidence_threshold: Minimum confidence for pseudo-labels
        """
        self.confidence_threshold = confidence_threshold
    
    def generate_labels(
        self,
        model: nn.Module,
        unlabeled_data: List[Dict],
        device: torch.device
    ) -> List[Dict]:
        """
        Generate pseudo-labels for unlabeled data.
        
        Args:
            model: Trained model
            unlabeled_data: Unlabeled data samples
            device: Device to run inference on
            
        Returns:
            List of data samples with pseudo-labels
        """
        model.eval()
        
        pseudo_labeled = []
        
        with torch.no_grad():
            for sample in unlabeled_data:
                points = sample['points'].to(device)
                
                # Get predictions
                outputs = model(points)
                
                # Extract boxes, scores, labels
                # TODO: Adapt based on actual model output format
                boxes = outputs.get('bboxes_3d')
                scores = outputs.get('scores')
                labels = outputs.get('labels')
                
                if boxes is None or scores is None:
                    continue
                
                # Filter by confidence
                mask = scores >= self.confidence_threshold
                
                if mask.sum() > 0:
                    pseudo_sample = {
                        'points': sample['points'],
                        'gt_bboxes_3d': boxes[mask].cpu().numpy(),
                        'gt_labels_3d': labels[mask].cpu().numpy() if labels is not None else None
                    }
                    pseudo_labeled.append(pseudo_sample)
        
        logger.info(
            f"Generated pseudo-labels for {len(pseudo_labeled)} samples "
            f"(confidence >= {self.confidence_threshold})"
        )
        
        return pseudo_labeled
