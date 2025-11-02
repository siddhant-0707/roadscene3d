"""Export PyTorch model to ONNX format."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_path: str,
    input_names: Optional[list] = None,
    output_names: Optional[list] = None,
    dynamic_axes: Optional[dict] = None,
    opset_version: int = 11
) -> str:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        sample_input: Sample input tensor for tracing
        output_path: Path to save ONNX model
        input_names: Optional input tensor names
        output_names: Optional output tensor names
        dynamic_axes: Optional dynamic axes specification
        opset_version: ONNX opset version
        
    Returns:
        Path to exported ONNX model
    """
    model.eval()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting model to ONNX: {output_path}")
    
    input_names = input_names or ['points']
    output_names = output_names or ['boxes', 'scores', 'labels']
    
    try:
        torch.onnx.export(
            model,
            sample_input,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            export_params=True,
            verbose=False,
            do_constant_folding=True
        )
        
        logger.info(f"Successfully exported ONNX model to {output_path}")
        return str(output_path)
    
    except Exception as e:
        logger.error(f"Failed to export ONNX model: {e}")
        raise


def validate_onnx_model(onnx_path: str) -> bool:
    """
    Validate exported ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        
    Returns:
        True if valid, False otherwise
    """
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        logger.info(f"ONNX model {onnx_path} is valid")
        return True
    except Exception as e:
        logger.error(f"ONNX model validation failed: {e}")
        return False
