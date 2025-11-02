"""Memory monitoring utilities for tracking VRAM and RAM usage during training."""

import psutil
import torch
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor GPU VRAM and system RAM usage."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize memory monitor.
        
        Args:
            device: PyTorch device (defaults to cuda:0 if available)
        """
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.has_cuda = torch.cuda.is_available() and self.device.type == 'cuda'
        
    def get_vram_usage(self) -> Dict[str, float]:
        """
        Get current GPU VRAM usage.
        
        Returns:
            Dictionary with 'allocated_mb', 'reserved_mb', 'total_mb', 'free_mb'
        """
        if not self.has_cuda:
            return {
                'allocated_mb': 0.0,
                'reserved_mb': 0.0,
                'total_mb': 0.0,
                'free_mb': 0.0,
                'usage_percent': 0.0
            }
        
        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)  # MB
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)  # MB
        total = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 2)  # MB
        free = total - reserved
        
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'total_mb': total,
            'free_mb': free,
            'usage_percent': (reserved / total) * 100.0
        }
    
    def get_ram_usage(self) -> Dict[str, float]:
        """
        Get current system RAM usage.
        
        Returns:
            Dictionary with 'used_mb', 'available_mb', 'total_mb', 'usage_percent'
        """
        mem = psutil.virtual_memory()
        return {
            'used_mb': mem.used / (1024 ** 2),
            'available_mb': mem.available / (1024 ** 2),
            'total_mb': mem.total / (1024 ** 2),
            'usage_percent': mem.percent
        }
    
    def get_all_usage(self) -> Dict[str, Dict[str, float]]:
        """
        Get both VRAM and RAM usage.
        
        Returns:
            Dictionary with 'vram' and 'ram' keys
        """
        return {
            'vram': self.get_vram_usage(),
            'ram': self.get_ram_usage()
        }
    
    def log_usage(self, stage: str = ""):
        """
        Log current memory usage.
        
        Args:
            stage: Optional stage identifier (e.g., "before_training", "after_epoch_1")
        """
        usage = self.get_all_usage()
        prefix = f"[{stage}] " if stage else ""
        
        if self.has_cuda:
            vram = usage['vram']
            logger.info(
                f"{prefix}VRAM: {vram['reserved_mb']:.1f}/{vram['total_mb']:.1f} MB "
                f"({vram['usage_percent']:.1f}%) - Allocated: {vram['allocated_mb']:.1f} MB"
            )
        
        ram = usage['ram']
        logger.info(
            f"{prefix}RAM: {ram['used_mb']:.1f}/{ram['total_mb']:.1f} MB "
            f"({ram['usage_percent']:.1f}%)"
        )
    
    def clear_cache(self):
        """Clear PyTorch CUDA cache to free VRAM."""
        if self.has_cuda:
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")


def check_memory_requirements(min_vram_gb: float = 6.0, min_ram_gb: float = 16.0) -> bool:
    """
    Check if system meets minimum memory requirements.
    
    Args:
        min_vram_gb: Minimum required VRAM in GB
        min_ram_gb: Minimum required RAM in GB
        
    Returns:
        True if requirements met, False otherwise
    """
    monitor = MemoryMonitor()
    usage = monitor.get_all_usage()
    
    vram_ok = True
    if monitor.has_cuda:
        vram_total_gb = usage['vram']['total_mb'] / 1024
        vram_ok = vram_total_gb >= min_vram_gb
        if not vram_ok:
            logger.warning(
                f"GPU VRAM insufficient: {vram_total_gb:.1f} GB < {min_vram_gb} GB required"
            )
    
    ram_total_gb = usage['ram']['total_mb'] / 1024
    ram_ok = ram_total_gb >= min_ram_gb
    if not ram_ok:
        logger.warning(
            f"RAM insufficient: {ram_total_gb:.1f} GB < {min_ram_gb} GB required"
        )
    
    return vram_ok and ram_ok
