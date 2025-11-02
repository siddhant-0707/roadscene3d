"""Contrastive self-supervised pretraining for LiDAR point clouds.

Implements PointContrast-style contrastive learning for 3D point clouds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for self-supervised learning."""
    
    def __init__(self, temperature: float = 0.1):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            features1: Features from view 1 [N, D]
            features2: Features from view 2 [N, D]
            labels: Optional correspondence labels [N]
            
        Returns:
            Contrastive loss scalar
        """
        # Normalize features
        features1 = F.normalize(features1, dim=-1)
        features2 = F.normalize(features2, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features1, features2.T) / self.temperature
        
        # Positive pairs are on diagonal (corresponding points)
        # Negative pairs are off-diagonal
        if labels is None:
            # Assume corresponding indices are positive pairs
            labels = torch.arange(features1.size(0), device=features1.device)
        
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class PointContrastPretrainer:
    """PointContrast-style pretraining for point clouds."""
    
    def __init__(
        self,
        encoder: nn.Module,
        projection_head: nn.Module,
        temperature: float = 0.1
    ):
        """
        Initialize pretrainer.
        
        Args:
            encoder: Point cloud encoder (e.g., PointNet)
            projection_head: Projection head for contrastive learning
            temperature: Temperature for contrastive loss
        """
        self.encoder = encoder
        self.projection_head = projection_head
        self.criterion = ContrastiveLoss(temperature=temperature)
    
    def create_views(
        self,
        points: torch.Tensor,
        augmentations: list
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create two augmented views of point cloud.
        
        Args:
            points: Input point cloud [N, 3]
            augmentations: List of augmentation functions
            
        Returns:
            Two augmented views
        """
        view1 = points.clone()
        view2 = points.clone()
        
        # Apply augmentations
        for aug in augmentations:
            view1 = aug(view1)
            view2 = aug(view2)
        
        return view1, view2
    
    def forward(
        self,
        points1: torch.Tensor,
        points2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for contrastive learning.
        
        Args:
            points1: First view [B, N, 3]
            points2: Second view [B, N, 3]
            
        Returns:
            Features for both views
        """
        # Encode both views
        features1 = self.encoder(points1)
        features2 = self.encoder(points2)
        
        # Project to embedding space
        proj1 = self.projection_head(features1)
        proj2 = self.projection_head(features2)
        
        return proj1, proj2
    
    def compute_loss(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            features1: Features from view 1
            features2: Features from view 2
            
        Returns:
            Loss value
        """
        return self.criterion(features1, features2)


def create_augmentations():
    """
    Create augmentation functions for contrastive learning.
    
    Returns:
        List of augmentation functions
    """
    def random_rotation(points):
        """Random rotation around z-axis."""
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=points.dtype, device=points.device)
        return torch.matmul(points, rotation.T)
    
    def random_scale(points):
        """Random scaling."""
        scale = np.random.uniform(0.8, 1.2)
        return points * scale
    
    def random_jitter(points):
        """Random noise jitter."""
        noise = torch.randn_like(points) * 0.01
        return points + noise
    
    return [random_rotation, random_scale, random_jitter]
