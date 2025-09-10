"""GPU optimization utilities for maximum RTX 4090 performance."""

import logging
import time
from typing import Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class GPUOptimizer:
    """Advanced GPU optimization for face processing on RTX 4090."""
    
    def __init__(self):
        self.cuda_context = None
        self.memory_pool = None
        self.stream = None
        self._initialize_cuda_context()
    
    def _initialize_cuda_context(self):
        """Initialize CUDA context for optimal memory management."""
        try:
            import torch
            if torch.cuda.is_available():
                # Set memory fraction for optimal allocation
                torch.cuda.set_per_process_memory_fraction(0.95, device=0)
                
                # Enable memory pool for faster allocations
                torch.cuda.empty_cache()
                
                # Create dedicated CUDA stream for face processing
                self.stream = torch.cuda.Stream()
                
                logger.info("CUDA context initialized for RTX 4090 optimization")
            
        except Exception as e:
            logger.warning(f"CUDA context initialization failed: {e}")
    
    def optimize_memory_layout(self, img_array: np.ndarray) -> np.ndarray:
        """Optimize memory layout for GPU processing."""
        try:
            # Ensure contiguous memory layout
            if not img_array.flags['C_CONTIGUOUS']:
                img_array = np.ascontiguousarray(img_array)
            
            # Align memory to 32-byte boundaries for optimal GPU access
            if img_array.nbytes % 32 != 0:
                padding = 32 - (img_array.nbytes % 32)
                padded_shape = list(img_array.shape)
                padded_shape[-1] += padding // img_array.itemsize
                padded_array = np.zeros(padded_shape, dtype=img_array.dtype)
                padded_array[..., :img_array.shape[-1]] = img_array
                img_array = padded_array
            
            return img_array
            
        except Exception as e:
            logger.warning(f"Memory layout optimization failed: {e}")
            return img_array
    
    def prefetch_to_gpu(self, data: np.ndarray) -> Optional[Any]:
        """Prefetch data to GPU memory for faster processing."""
        try:
            import torch
            if torch.cuda.is_available() and self.stream:
                with torch.cuda.stream(self.stream):
                    tensor = torch.from_numpy(data).cuda(non_blocking=True)
                    return tensor
        except Exception as e:
            logger.debug(f"GPU prefetch failed: {e}")
        return None
    
    def get_optimal_batch_size(self, image_size: tuple) -> int:
        """Calculate optimal batch size based on image dimensions and GPU memory."""
        try:
            import torch
            if not torch.cuda.is_available():
                return 1
            
            # Get available GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = gpu_memory * 0.8  # Use 80% of total memory
            
            # Estimate memory per image (rough calculation)
            width, height = image_size
            channels = 3
            bytes_per_pixel = 4  # float32
            
            # Account for intermediate activations (multiply by 4 for safety)
            memory_per_image = width * height * channels * bytes_per_pixel * 4
            
            optimal_batch = max(1, int(available_memory // memory_per_image))
            
            # Cap at reasonable maximum for face detection
            return min(optimal_batch, 16)
            
        except Exception as e:
            logger.warning(f"Batch size optimization failed: {e}")
            return 8  # Default for RTX 4090
    
    def enable_mixed_precision(self) -> Dict[str, Any]:
        """Enable mixed precision training optimizations."""
        try:
            import torch
            if torch.cuda.is_available():
                # Enable TensorFloat-32 for RTX 4090
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Enable optimized attention for transformer models
                torch.backends.cuda.enable_flash_sdp(True)
                
                return {
                    'tf32_enabled': True,
                    'flash_attention': True,
                    'mixed_precision': True
                }
        except Exception as e:
            logger.warning(f"Mixed precision setup failed: {e}")
        
        return {'mixed_precision': False}
    
    def warm_up_gpu(self, dummy_shape: tuple = (640, 640, 3)):
        """Warm up GPU with dummy operations to optimize kernel loading."""
        try:
            import torch
            if torch.cuda.is_available():
                # Create dummy tensor and perform operations
                dummy_data = torch.randn(dummy_shape, device='cuda')
                
                # Perform typical operations used in face detection
                _ = torch.nn.functional.conv2d(
                    dummy_data.unsqueeze(0).permute(0, 3, 1, 2),
                    torch.randn(64, 3, 3, 3, device='cuda')
                )
                
                # Synchronize to ensure operations complete
                torch.cuda.synchronize()
                
                logger.info("GPU warmup completed")
                
        except Exception as e:
            logger.warning(f"GPU warmup failed: {e}")
    
    def optimize_cudnn(self):
        """Optimize cuDNN settings for RTX 4090."""
        try:
            import torch
            if torch.cuda.is_available():
                # Enable cuDNN benchmark mode for consistent input sizes
                torch.backends.cudnn.benchmark = True
                
                # Enable deterministic mode for reproducible results
                torch.backends.cudnn.deterministic = False  # Faster but non-deterministic
                
                # Enable cuDNN auto-tuner
                torch.backends.cudnn.enabled = True
                
                logger.info("cuDNN optimizations enabled")
                
        except Exception as e:
            logger.warning(f"cuDNN optimization failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current GPU performance statistics."""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                memory_allocated = torch.cuda.memory_allocated(0)
                memory_reserved = torch.cuda.memory_reserved(0)
                
                return {
                    'device_name': props.name,
                    'total_memory_gb': props.total_memory / (1024**3),
                    'memory_allocated_gb': memory_allocated / (1024**3),
                    'memory_reserved_gb': memory_reserved / (1024**3),
                    'memory_utilization': memory_allocated / props.total_memory,
                    'compute_capability': f"{props.major}.{props.minor}",
                    'multiprocessor_count': props.multi_processor_count
                }
        except Exception as e:
            logger.warning(f"Performance stats collection failed: {e}")
        
        return {}


# Global GPU optimizer instance
gpu_optimizer = GPUOptimizer()
