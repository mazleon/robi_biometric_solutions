"""Optimized Configuration for FAISS GPU service with performance enhancements."""

import os
from typing import Optional


class OptimizedFaissConfig:
    """Optimized configuration class for FAISS GPU service."""
    
    # Vector settings
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "512"))
    
    # GPU settings with optimizations
    GPU_DEVICE_ID: int = int(os.getenv("GPU_DEVICE_ID", "0"))
    GPU_MEMORY_GB: int = int(os.getenv("GPU_MEMORY_GB", "16"))  # RTX 4090 allocation
    USE_FP16: bool = os.getenv("USE_FP16", "true").lower() == "true"
    
    # Index settings with performance options
    INDEX_TYPE: str = os.getenv("INDEX_TYPE", "IndexFlatIP")  # Options: IndexFlatIP, IndexIVFFlat, IndexHNSW
    INDEX_PATH: str = os.getenv("INDEX_PATH", "/app/data/faiss_index.bin")
    METADATA_PATH: str = os.getenv("METADATA_PATH", "/app/data/metadata.json")
    
    # Performance settings optimized
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "64"))  # Increased for better GPU utilization
    MAX_VECTORS: int = int(os.getenv("MAX_VECTORS", "1000000"))
    
    # Search optimization settings
    SEARCH_NPROBE: int = int(os.getenv("SEARCH_NPROBE", "32"))  # For IVF indices
    HNSW_EF_CONSTRUCTION: int = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))
    HNSW_EF_SEARCH: int = int(os.getenv("HNSW_EF_SEARCH", "50"))
    HNSW_M: int = int(os.getenv("HNSW_M", "32"))
    
    # Memory optimization
    USE_MEMORY_MAPPING: bool = os.getenv("USE_MEMORY_MAPPING", "true").lower() == "true"
    SAVE_INTERVAL_SECONDS: int = int(os.getenv("SAVE_INTERVAL_SECONDS", "30"))
    
    # API settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Health check settings
    HEALTH_CHECK_INTERVAL: int = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
    
    # Performance monitoring
    ENABLE_PERFORMANCE_METRICS: bool = os.getenv("ENABLE_PERFORMANCE_METRICS", "true").lower() == "true"
    METRICS_COLLECTION_INTERVAL: int = int(os.getenv("METRICS_COLLECTION_INTERVAL", "60"))


# Global optimized config instance
optimized_config = OptimizedFaissConfig()
