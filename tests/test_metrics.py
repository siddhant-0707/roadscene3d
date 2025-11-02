"""Tests for evaluation metrics."""

import pytest
import numpy as np
import torch


def test_compute_map():
    """Test mAP computation."""
    from src.evaluation.metrics import compute_map
    
    # Create dummy predictions and ground truth
    pred_boxes = [
        np.array([[10, 10, 0, 2, 2, 2, 0]], dtype=np.float32)
    ]
    pred_scores = [np.array([0.9])]
    pred_labels = [np.array([0])]
    
    gt_boxes = [
        np.array([[10, 10, 0, 2, 2, 2, 0]], dtype=np.float32)
    ]
    gt_labels = [np.array([0])]
    
    # Compute mAP
    metrics = compute_map(
        pred_boxes,
        pred_scores,
        pred_labels,
        gt_boxes,
        gt_labels,
        iou_thresholds=[0.5, 0.7]
    )
    
    # Check that metrics dictionary is returned
    assert isinstance(metrics, dict)
    assert 'mAP@0.5' in metrics or 'mAP@0.7' in metrics


def test_memory_monitor():
    """Test memory monitoring."""
    from src.utils.memory_monitor import MemoryMonitor
    
    monitor = MemoryMonitor()
    
    # Test getting usage
    usage = monitor.get_all_usage()
    assert 'vram' in usage
    assert 'ram' in usage
    
    # Test logging (should not raise)
    monitor.log_usage("test")


def test_model_registry():
    """Test model registry."""
    from src.models.registry import ModelRegistry
    
    registry = ModelRegistry()
    
    # Should not raise
    latest_version = registry.get_latest_version()
    assert isinstance(latest_version, int)


@pytest.mark.skip(reason="Requires actual model")
def test_inference_latency():
    """Test inference latency measurement."""
    from src.evaluation.metrics import measure_inference_latency
    import torch.nn as nn
    
    # Dummy model
    model = nn.Linear(10, 1)
    sample_input = torch.randn(1, 10)
    
    metrics = measure_inference_latency(model, sample_input, num_runs=10)
    
    assert 'latency_mean_ms' in metrics
    assert 'throughput_fps' in metrics
