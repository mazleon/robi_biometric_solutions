"""
GPU configuration utilities for RTX 4090 optimization.
"""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information and capabilities."""
    gpu_info = {
        'gpu_available': False,
        'gpu_count': 0,
        'gpu_name': 'Unknown',
        'gpu_memory_gb': 0,
        'cuda_version': 'Unknown',
        'compute_capability': 'Unknown'
    }
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_info['gpu_available'] = True
            gpu_info['gpu_count'] = torch.cuda.device_count()
            
            if gpu_info['gpu_count'] > 0:
                gpu_info['gpu_name'] = torch.cuda.get_device_name(0)
                gpu_properties = torch.cuda.get_device_properties(0)
                gpu_info['gpu_memory_gb'] = gpu_properties.total_memory / (1024**3)
                gpu_info['compute_capability'] = f"{gpu_properties.major}.{gpu_properties.minor}"
                
                # Get CUDA version
                try:
                    gpu_info['cuda_version'] = torch.version.cuda
                except:
                    gpu_info['cuda_version'] = 'Unknown'
        
        logger.info(f"GPU Info: {gpu_info}")
        return gpu_info
        
    except ImportError:
        logger.warning("PyTorch not available, cannot get GPU info")
        return gpu_info
    except Exception as e:
        logger.error(f"Error getting GPU info: {e}")
        return gpu_info


def apply_gpu_config() -> bool:
    """Apply GPU-specific configuration for RTX 4090."""
    try:
        logger.info("Applying GPU configuration for RTX 4090...")
        
        # CUDA Environment Variables
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async execution
        os.environ['CUDA_CACHE_DISABLE'] = '0'    # Enable caching
        
        # PyTorch optimizations
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048,expandable_segments:True'
        
        # ONNX Runtime GPU optimizations (disable aggressive features)
        os.environ['ORT_TENSORRT_MAX_WORKSPACE_SIZE'] = str(2 * 1024 * 1024 * 1024)  # 2GB (reduced)
        os.environ['ORT_TENSORRT_FP16_ENABLE'] = '0'  # Disable FP16 for compatibility
        os.environ['ORT_TENSORRT_ENGINE_CACHE_ENABLE'] = '0'  # Disable engine cache
        
        # cuDNN optimizations
        os.environ['CUDNN_BENCHMARK'] = '1'
        os.environ['CUDNN_DETERMINISTIC'] = '0'
        
        # Memory management
        os.environ['PYTORCH_CUDA_MEMORY_FRACTION'] = '0.95'
        
        logger.info("GPU configuration applied successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply GPU configuration: {e}")
        return False


def optimize_torch_settings():
    """Apply PyTorch-specific optimizations."""
    try:
        import torch
        
        if torch.cuda.is_available():
            # Enable TensorFloat-32 for RTX 4090
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cuDNN benchmarking
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.95, device=0)
            
            logger.info("PyTorch GPU optimizations applied")
            return True
        else:
            logger.warning("CUDA not available, skipping PyTorch optimizations")
            return False
            
    except ImportError:
        logger.warning("PyTorch not available")
        return False
    except Exception as e:
        logger.error(f"Failed to apply PyTorch optimizations: {e}")
        return False


def check_gpu_compatibility() -> bool:
    """Check if GPU is compatible with the application."""
    try:
        gpu_info = get_gpu_info()
        
        if not gpu_info['gpu_available']:
            logger.warning("No GPU available")
            return False
        
        # Check minimum memory requirement (8GB)
        if gpu_info['gpu_memory_gb'] < 8:
            logger.warning(f"GPU memory ({gpu_info['gpu_memory_gb']:.1f}GB) below recommended 8GB")
            return False
        
        # Check compute capability (minimum 7.0 for RTX series)
        try:
            major, minor = gpu_info['compute_capability'].split('.')
            compute_version = float(f"{major}.{minor}")
            if compute_version < 7.0:
                logger.warning(f"GPU compute capability ({compute_version}) below recommended 7.0")
                return False
        except:
            logger.warning("Could not determine compute capability")
        
        logger.info("GPU compatibility check passed")
        return True
        
    except Exception as e:
        logger.error(f"GPU compatibility check failed: {e}")
        return False


def setup_faiss_gpu():
    """Setup FAISS GPU configuration."""
    try:
        import faiss
        
        # Check if GPU FAISS is available
        if hasattr(faiss, 'StandardGpuResources'):
            gpu_info = get_gpu_info()
            if gpu_info['gpu_available']:
                logger.info("FAISS GPU support available")
                return True
        
        logger.warning("FAISS GPU support not available")
        return False
        
    except ImportError:
        logger.warning("FAISS not available")
        return False
    except Exception as e:
        logger.error(f"FAISS GPU setup failed: {e}")
        return False


def get_optimal_batch_size() -> int:
    """Get optimal batch size based on GPU memory."""
    try:
        gpu_info = get_gpu_info()
        memory_gb = gpu_info['gpu_memory_gb']
        
        # RTX 4090 (24GB) optimizations
        if memory_gb >= 20:
            return 32
        elif memory_gb >= 16:
            return 24
        elif memory_gb >= 12:
            return 16
        elif memory_gb >= 8:
            return 12
        else:
            return 8
            
    except Exception as e:
        logger.error(f"Failed to determine optimal batch size: {e}")
        return 8


def monitor_gpu_memory():
    """Monitor GPU memory usage."""
    try:
        import torch
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'utilization_percent': (allocated / total) * 100
            }
        else:
            return None
            
    except Exception as e:
        logger.error(f"Failed to monitor GPU memory: {e}")
        return None
