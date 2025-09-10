"""Configuration for FAISS GPU service."""

import os
from typing import Optional


class FaissConfig:
    """Configuration class for FAISS GPU service."""
    
    # Vector settings
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "512"))
    
    # GPU settings
    GPU_DEVICE_ID: int = int(os.getenv("GPU_DEVICE_ID", "0"))
    GPU_MEMORY_GB: int = int(os.getenv("GPU_MEMORY_GB", "16"))  # RTX 4090 allocation
    USE_FP16: bool = os.getenv("USE_FP16", "true").lower() == "true"
    
    # Index settings
    INDEX_TYPE: str = os.getenv("INDEX_TYPE", "IndexFlatIP")  # Cosine similarity
    INDEX_PATH: str = os.getenv("INDEX_PATH", "/app/data/faiss_index.bin")
    METADATA_PATH: str = os.getenv("METADATA_PATH", "/app/data/metadata.json")
    
    # Performance settings
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    MAX_VECTORS: int = int(os.getenv("MAX_VECTORS", "1000000"))
    
    # API settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Health check settings
    HEALTH_CHECK_INTERVAL: int = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))


# Global config instance
config = FaissConfig()
