"""Optimized configuration settings for the face verification system."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """High-performance application settings with GPU optimization."""
    
    # Application configuration
    app_name: str = Field(default="Face Verification API", env="APP_NAME")
    
    # Server configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Qdrant service configuration (Docker container on ports 6333-6334)
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_grpc_port: int = Field(default=6334, env="QDRANT_GRPC_PORT")
    qdrant_collection: str = Field(default="face_embeddings", env="QDRANT_COLLECTION")
    qdrant_timeout: int = Field(default=30, env="QDRANT_TIMEOUT")
    qdrant_retries: int = Field(default=3, env="QDRANT_RETRIES")
    
    # FAISS configuration for hybrid search
    faiss_enabled: bool = Field(default=True, env="FAISS_ENABLED")
    faiss_index_type: str = Field(default="IVF", env="FAISS_INDEX_TYPE")  # IVF, Flat, HNSW
    faiss_nlist: int = Field(default=4096, env="FAISS_NLIST")  # Number of clusters
    faiss_nprobe: int = Field(default=64, env="FAISS_NPROBE")  # Search clusters
    faiss_cache_size: int = Field(default=100000, env="FAISS_CACHE_SIZE")  # Hot vectors in FAISS
    
    # Face processing configuration
    model_name: str = Field(default="buffalo_l", env="MODEL_NAME", description="Primary recognition model.")
    # Pluggable detector model. If None, uses the detector bundled with `model_name`.
    # Example: 'scrfd_10g_bnkps' for a more powerful detector.
    detector_model_name: Optional[str] = Field(default=None, env="DETECTOR_MODEL_NAME")
    detection_size: tuple = Field(default=(640, 640), description="Input size for the detector model.")
    similarity_threshold: float = Field(default=0.45, env="SIMILARITY_THRESHOLD")
    max_faces_per_image: int = Field(default=10, env="MAX_FACES_PER_IMAGE", description="Max faces to process per image.")
    max_image_size: int = Field(default=1600, env="MAX_IMAGE_SIZE", description="Max image dimension before resizing.")
    
    # Expert-level detector configuration
    detection_threshold: float = Field(default=0.30, env="DETECTION_THRESHOLD", description="Primary confidence threshold for face detection.")
    detector_nms_threshold: float = Field(default=0.4, env="NMS_THRESHOLD", description="Non-Maximum Suppression threshold.")
    fallback_detection_threshold: float = Field(default=0.20, env="FALLBACK_DETECTION_THRESHOLD", description="Secondary threshold for fallback strategies.")
    
    # Vector processing configuration
    embedding_dimension: int = Field(default=512, env="EMBEDDING_DIMENSION")
    cosine_similarity_threshold: float = Field(default=0.65, env="COSINE_SIMILARITY_THRESHOLD")
    
    # GPU optimization configuration
    use_gpu: bool = Field(default=True, env="USE_GPU")
    gpu_device_id: int = Field(default=0, env="GPU_DEVICE_ID")
    ctx_id: int = Field(default=0, env="CTX_ID")
    gpu_memory_fraction: float = Field(default=0.90, env="GPU_MEMORY_FRACTION", description="Fraction of GPU VRAM to allocate.")
    aggressive_gpu_optimizations: bool = Field(default=False, env="AGGRESSIVE_GPU_OPTIMIZATIONS", description="Enable experimental GPU optimizations like CUDA Graphs.")
    batch_size: int = Field(default=1, env="BATCH_SIZE", description="Batch size for recognition (detector is always batch 1).")
    onnx_providers: List[str] = Field(
        default=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )  # Remove env to avoid JSON parsing issues
    
    # Legacy Qdrant service configuration (deprecated - use qdrant_url instead)
    qdrant_service_url: str = Field(default="http://localhost:6333", env="QDRANT_SERVICE_URL")
    
    # File upload configuration (optimized limits)
    max_file_size: int = Field(default=5 * 1024 * 1024, env="MAX_FILE_SIZE")  # 5MB for faster processing
    allowed_extensions: List[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"]
    )  # Remove env to avoid JSON parsing issues
    
    # Logging configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default="logs/app.log", env="LOG_FILE")
    structured_logging: bool = Field(default=True, env="STRUCTURED_LOGGING")
    
    # Performance configuration (RTX 4090 optimized)
    max_workers: int = Field(default=8, env="MAX_WORKERS")  # Increased for RTX 4090
    request_timeout: int = Field(default=15, env="REQUEST_TIMEOUT")  # Reduced for faster responses
    batch_size: int = Field(default=16, env="BATCH_SIZE")  # Optimal for RTX 4090
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    cache_size: int = Field(default=1000, env="CACHE_SIZE")
    
    # Connection pooling
    max_connections: int = Field(default=100, env="MAX_CONNECTIONS")
    connection_timeout: int = Field(default=5, env="CONNECTION_TIMEOUT")
    
    # Security settings
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    cors_origins: List[str] = Field(default_factory=lambda: ['*'])  # Remove env to avoid JSON parsing

    model_config = SettingsConfigDict(
        env_file=".env", 
        case_sensitive=False, 
        str_strip_whitespace=True,
        extra="ignore"  # Ignore extra fields instead of raising validation errors
    )


    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._create_directories()
        self._configure_runtime()

    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        Path("logs").mkdir(exist_ok=True)
        Path("data").mkdir(exist_ok=True)

    def _configure_runtime(self):
        """Configure runtime optimizations for RTX 4090."""
        import logging
        logger = logging.getLogger(__name__)

        if self.use_gpu:
            # CUDA optimizations for RTX 4090
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_device_id)
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            
            # Memory optimizations
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:2048'
            
            # TensorFloat-32 for RTX 4090
            os.environ['NVIDIA_TF32_OVERRIDE'] = '1'
            
            # ONNX Runtime optimizations
            os.environ['ORT_TENSORRT_MAX_WORKSPACE_SIZE'] = str(8 * 1024 * 1024 * 1024)  # 8GB
            os.environ['ORT_TENSORRT_FP16_ENABLE'] = '1'
            
            try:
                import onnxruntime as ort
                available_providers = ort.get_available_providers()
                logger.info(f"Available ONNX providers: {available_providers}")
                
                if 'CUDAExecutionProvider' in available_providers:
                    self.onnx_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    self.ctx_id = self.gpu_device_id
                    logger.info(f"GPU acceleration enabled on device {self.gpu_device_id}")
                else:
                    logger.warning("GPU requested but CUDA provider not available. Falling back to CPU.")
                    self.use_gpu = False
                    # GPU fallback configured
                    self.ctx_id = -1
            except ImportError as e:
                logger.warning(f"ONNX Runtime import failed: {e}. Falling back to CPU.")
                self.use_gpu = False
                # GPU fallback configured
                self.ctx_id = -1
            except Exception as e:
                logger.warning(f"GPU configuration failed: {e}. Falling back to CPU.")
                self.use_gpu = False
                # GPU fallback configured
                self.ctx_id = -1
        
        if not self.use_gpu:
            self.onnx_providers = ['CPUExecutionProvider']
            self.ctx_id = -1
            logger.info("CPU processing mode enabled.")

    def get_runtime_info(self) -> dict[str, Any]:
        """Get current runtime configuration information."""
        return {
            "use_gpu": self.use_gpu,
            "gpu_device_id": self.gpu_device_id,
            "ctx_id": self.ctx_id,
            "onnx_providers": self.onnx_providers,
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "qdrant_enabled": True,
            "max_image_size": self.max_image_size
        }
    
    @property
    def embedding_dim(self) -> int:
        """Face embedding dimension (512 for InsightFace)."""
        return 512
    
    @property
    def cosine_threshold(self) -> float:
        """Cosine similarity threshold for face verification."""
        return self.similarity_threshold


# Global settings instance
settings = Settings()
