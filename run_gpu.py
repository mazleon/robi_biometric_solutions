#!/usr/bin/env python3
"""
GPU-accelerated startup script for Face Verification API
Optimized for NVIDIA RTX 4090 (24GB VRAM)
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Import GPU configuration from src
from src.utils.gpu_config import apply_gpu_config, get_gpu_info

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_gpu_requirements():
    """Check if GPU requirements are met."""
    logger.info("Checking GPU requirements...")
    
    # Check CUDA availability
    try:
        import torch
        if not torch.cuda.is_available():
            logger.error("CUDA is not available. Please install CUDA 12.1 or compatible version.")
            return False
        
        gpu_info = get_gpu_info()
        logger.info(f"GPU Info: {gpu_info}")
        
        if gpu_info['gpu_memory_gb'] < 8:
            logger.warning(f"GPU has only {gpu_info['gpu_memory_gb']}GB VRAM. Recommended: 8GB+")
        
        return True
        
    except ImportError:
        logger.error("PyTorch not installed. Please install PyTorch with CUDA support.")
        return False

def check_dependencies():
    """Check if all GPU dependencies are installed."""
    logger.info("Checking GPU dependencies...")
    
    # Check core packages with proper import handling
    packages_status = {}
    
    # Check PyTorch
    try:
        import torch
        packages_status['torch'] = True
        logger.info("✓ torch is installed")
    except ImportError:
        packages_status['torch'] = False
        logger.error("✗ torch is missing")
    
    # Check ONNX Runtime (GPU or CPU)
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            packages_status['onnxruntime-gpu'] = True
            logger.info("✓ onnxruntime-gpu is available")
        else:
            packages_status['onnxruntime-gpu'] = False
            logger.info("✓ onnxruntime (CPU) is installed, GPU provider not available")
    except ImportError:
        packages_status['onnxruntime-gpu'] = False
        logger.error("✗ onnxruntime is missing")
    
    # Check Qdrant service connectivity (updated port)
    try:
        import requests
        response = requests.get("http://localhost:6333/health", timeout=5)
        if response.status_code == 200:
            packages_status['qdrant-service'] = True
            logger.info("✓ Qdrant service is available at localhost:6333")
        else:
            packages_status['qdrant-service'] = False
            logger.warning("✗ Qdrant service not responding properly")
    except Exception:
        packages_status['qdrant-service'] = False
        logger.warning("✗ Qdrant service not available at localhost:6333")
    
    # Check InsightFace
    try:
        import insightface
        packages_status['insightface'] = True
        logger.info("✓ insightface is installed")
    except ImportError:
        packages_status['insightface'] = False
        logger.error("✗ insightface is missing")
    
    # Check if critical packages are available
    critical_missing = []
    if not packages_status.get('torch', False):
        critical_missing.append('torch')
    if not packages_status.get('insightface', False):
        critical_missing.append('insightface')
    
    if critical_missing:
        logger.error(f"Critical packages missing: {critical_missing}")
        return False
    
    # Warn about GPU optimizations if not available
    if not packages_status.get('onnxruntime-gpu', False):
        logger.warning("ONNX Runtime GPU not available, falling back to CPU")
    if not packages_status.get('qdrant-service', False):
        logger.warning("Qdrant service not available, vector operations may fail")
    
    return True

def setup_gpu_environment():
    """Setup GPU environment variables and configuration."""
    logger.info("Setting up GPU environment for RTX 4090...")
    
    # Apply GPU configuration
    apply_gpu_config()
    
    # Additional CUDA optimizations for RTX 4090
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async execution
    os.environ['CUDA_CACHE_DISABLE'] = '0'    # Enable caching
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048'  # Memory management
    
    # ONNX Runtime optimizations
    os.environ['ORT_TENSORRT_MAX_WORKSPACE_SIZE'] = str(8 * 1024 * 1024 * 1024)  # 8GB
    os.environ['ORT_TENSORRT_FP16_ENABLE'] = '1'  # Enable FP16
    
    logger.info("GPU environment configured successfully")

def create_directories():
    """Create necessary directories."""
    directories = ['data', 'logs', 'models']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")

def run_gpu_tests():
    """Run basic GPU functionality tests."""
    logger.info("Running GPU functionality tests...")
    
    try:
        # Test CUDA
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"✓ CUDA test passed - Device: {device}, Memory: {memory:.1f}GB")
        
        # Test ONNX Runtime GPU
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            logger.info("✓ ONNX Runtime GPU support available")
        else:
            logger.warning("✗ ONNX Runtime GPU support not available")
        
        # Test Qdrant service
        import requests
        try:
            response = requests.get("http://localhost:6333/health", timeout=5)
            if response.status_code == 200:
                logger.info("✓ Qdrant service available")
            else:
                logger.warning("✗ Qdrant service not responding")
        except:
            logger.warning("✗ Qdrant service not available")
        
        return True
        
    except Exception as e:
        logger.error(f"GPU tests failed: {e}")
        return False

def start_application():
    """Start the face verification application with GPU acceleration."""
    logger.info("Starting Face Verification API with GPU acceleration...")
    
    try:
        # Start the application
        cmd = [
            sys.executable, "-m", "uvicorn",
            "src.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--workers", "1",  # Single worker for GPU memory management
            "--log-level", "info"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Application stopped by user")

def main():
    """Main function to run GPU-accelerated face verification."""
    logger.info("=" * 60)
    logger.info("Face Verification API - GPU Accelerated Mode")
    logger.info("Optimized for NVIDIA RTX 4090 with Qdrant Vector Database")
    logger.info("=" * 60)
    
    # Check requirements
    if not check_gpu_requirements():
        logger.error("GPU requirements not met. Exiting.")
        sys.exit(1)
    
    if not check_dependencies():
        logger.error("Dependencies not satisfied. Exiting.")
        sys.exit(1)
    
    # Setup environment
    setup_gpu_environment()
    create_directories()
    
    # Run tests
    if not run_gpu_tests():
        logger.warning("Some GPU tests failed. Continuing anyway...")
    
    # Start application
    logger.info("All checks passed. Starting GPU-accelerated application...")
    start_application()

if __name__ == "__main__":
    main()
