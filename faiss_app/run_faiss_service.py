#!/usr/bin/env python3
"""Standalone script to run FAISS GPU service."""

import os
import sys
import logging
import uvicorn
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Main function to run FAISS GPU service."""
    logger.info("Starting FAISS GPU Microservice...")
    logger.info(f"Configuration:")
    logger.info(f"  - Host: {config.HOST}:{config.PORT}")
    logger.info(f"  - GPU Device: {config.GPU_DEVICE_ID}")
    logger.info(f"  - GPU Memory: {config.GPU_MEMORY_GB}GB")
    logger.info(f"  - FP16: {config.USE_FP16}")
    logger.info(f"  - Index Type: {config.INDEX_TYPE}")
    logger.info(f"  - Embedding Dim: {config.EMBEDDING_DIM}")
    
    # Create data directory
    os.makedirs(os.path.dirname(config.INDEX_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(config.METADATA_PATH), exist_ok=True)
    
    try:
        uvicorn.run(
            "main:app",
            host=config.HOST,
            port=config.PORT,
            log_level=config.LOG_LEVEL.lower(),
            reload=False,
            access_log=True
        )
    except Exception as e:
        logger.error(f"Failed to start FAISS service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
