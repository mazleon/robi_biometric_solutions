"""Optimized face processing module with GPU acceleration and memory management."""

import io
import hashlib
import logging
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from PIL import Image
import insightface

from src.config.settings import settings
from src.core.gpu_optimizer import gpu_optimizer

logger = logging.getLogger(__name__)


class OptimizedFaceProcessor:
    """High-performance face processor with GPU optimization and memory pooling."""
    
    def __init__(self):
        self.face_app = None
        self._model_warmed_up = False
        self._embedding_cache = {}  # Simple LRU cache for embeddings
        self._cache_size = 1000
        self._initialize_models()
        self._warmup_models()
    
    def _initialize_models(self):
        """Initialize InsightFace models with optimized GPU configuration."""
        try:
            import onnxruntime as ort
            ort.set_default_logger_severity(3)
            
            # GPU-optimized provider configuration
            if settings.use_gpu and 'CUDAExecutionProvider' in settings.onnx_providers:
                gpu_memory_limit = int(24 * 1024 * 1024 * 1024 * settings.gpu_memory_fraction)
                provider_options = [{
                    'device_id': settings.gpu_device_id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': gpu_memory_limit,
                    'cudnn_conv_algo_search': 'HEURISTIC',
                    'do_copy_in_default_stream': True,
                    'enable_cuda_graph': True,
                    'use_tf32': True,
                }]
                providers = list(zip(settings.onnx_providers, provider_options + [{}]))
            else:
                providers = settings.onnx_providers
            
            self.face_app = insightface.app.FaceAnalysis(
                name=settings.model_name,
                allowed_modules=['detection', 'recognition'],
                providers=providers
            )
            
            self.face_app.prepare(
                ctx_id=settings.ctx_id,
                det_size=settings.detection_size
            )
            
            logger.info(f"Optimized face processor initialized with {settings.get_runtime_info()}")
            self._verify_gpu_usage()
            
        except Exception as e:
            logger.error(f"Failed to initialize face processing models: {e}")
            raise
    
    def _warmup_models(self):
        """GPU warmup with memory pre-allocation."""
        if self._model_warmed_up or not self.face_app:
            return
            
        try:
            logger.info("Warming up models with GPU optimizations...")
            gpu_optimizer.optimize_cudnn()
            gpu_optimizer.enable_mixed_precision()
            
            # Optimized warmup with common image sizes
            warmup_sizes = [(640, 640, 3), (1024, 1024, 3)]
            for size in warmup_sizes:
                dummy_image = np.random.randint(0, 255, size, dtype=np.uint8)
                dummy_image = gpu_optimizer.optimize_memory_layout(dummy_image)
                gpu_optimizer.prefetch_to_gpu(dummy_image)
                _ = self.face_app.get(dummy_image)
            
            gpu_optimizer.warm_up_gpu()
            self._model_warmed_up = True
            
            perf_stats = gpu_optimizer.get_performance_stats()
            logger.info(f"Models warmed up successfully: {perf_stats}")
            
        except Exception as e:
            logger.warning(f"Model warmup failed (non-critical): {e}")
            self._model_warmed_up = True
    
    def _verify_gpu_usage(self):
        """Verify GPU utilization."""
        try:
            import torch
            if torch.cuda.is_available() and settings.use_gpu:
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated(0)
                
                dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                _ = self.face_app.get(dummy_image)
                
                final_memory = torch.cuda.memory_allocated(0)
                memory_used_mb = (final_memory - initial_memory) / (1024 * 1024)
                
                if memory_used_mb > 10:
                    logger.info(f"✓ GPU verified - Memory used: {memory_used_mb:.1f}MB")
                else:
                    logger.warning(f"⚠ GPU may not be active - Memory used: {memory_used_mb:.1f}MB")
                    
        except Exception as e:
            logger.warning(f"GPU verification failed: {e}")
    
    def get_multiple_embeddings(self, image: Image.Image) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """Extract embeddings with GPU optimization and caching."""
        try:
            # Fast validation
            width, height = image.size
            if width < 50 or height < 50 or width > 4000 or height > 4000:
                logger.warning(f"Invalid image size: {width}x{height}")
                return []

            # Check cache first
            image_hash = self._calculate_image_hash(image)
            if image_hash in self._embedding_cache:
                logger.debug("Cache hit for image embedding")
                return self._embedding_cache[image_hash]

            # GPU-optimized preprocessing
            img_array = self._preprocess_image_optimized(image)
            
            if settings.use_gpu:
                img_array = gpu_optimizer.optimize_memory_layout(img_array)
                gpu_optimizer.prefetch_to_gpu(img_array)
            
            # Optimized face detection
            faces = self.face_app.get(img_array, max_num=5)
            if not faces:
                return []

            embeddings_with_info = []
            for i, face in enumerate(faces):
                if face.embedding is None:
                    continue

                # Optimized normalization
                embedding = face.embedding.astype('float32')
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding /= norm

                # Minimal face info
                bbox = face.bbox
                face_info = {
                    'index': i,
                    'bbox': bbox.tolist(),
                    'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    'confidence': float(face.det_score)
                }
                embeddings_with_info.append((embedding, face_info))

            # Sort by area and cache result
            if len(embeddings_with_info) > 1:
                embeddings_with_info.sort(key=lambda x: x[1]['area'], reverse=True)
            
            # Cache management
            self._cache_embedding(image_hash, embeddings_with_info)
            
            return embeddings_with_info

        except Exception as e:
            logger.error(f"Failed to extract embeddings: {e}")
            return []
    
    def _preprocess_image_optimized(self, image: Image.Image) -> np.ndarray:
        """Optimized image preprocessing with memory efficiency."""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Optimized resizing
            width, height = image.size
            max_size = min(settings.max_image_size, 1024)
            
            if max(width, height) > max_size:
                ratio = max_size / max(width, height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
            
            # Direct numpy conversion with memory optimization
            img_array = np.ascontiguousarray(np.array(image, dtype=np.uint8))
            
            # Shape validation and correction
            if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif len(img_array.shape) == 3 and img_array.shape[2] > 3:
                    img_array = img_array[:, :, :3]
                else:
                    raise ValueError(f"Unsupported image shape: {img_array.shape}")
            
            return img_array
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return np.ascontiguousarray(np.array(image, dtype=np.uint8))
    
    def _calculate_image_hash(self, image: Image.Image) -> str:
        """Calculate hash for image caching."""
        try:
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='JPEG', quality=85)
            return hashlib.md5(img_bytes.getvalue()).hexdigest()
        except Exception:
            return str(hash(str(image.size)))
    
    def _cache_embedding(self, image_hash: str, embeddings: List[Tuple[np.ndarray, Dict[str, Any]]]):
        """Cache embeddings with LRU eviction."""
        if len(self._embedding_cache) >= self._cache_size:
            # Simple LRU: remove oldest entry
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
        
        self._embedding_cache[image_hash] = embeddings
    
    def calculate_embedding_hash(self, embedding: np.ndarray) -> str:
        """Calculate hash of embedding for deduplication."""
        try:
            return hashlib.sha256(embedding.tobytes()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate embedding hash: {e}")
            return ""
    
    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Optimized cosine similarity calculation."""
        try:
            # Vectorized normalization and dot product
            emb1_norm = emb1 / np.linalg.norm(emb1)
            emb2_norm = emb2 / np.linalg.norm(emb2)
            return float(np.dot(emb1_norm, emb2_norm))
        except Exception as e:
            logger.error(f"Failed to compare embeddings: {e}")
            return 0.0
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")


# Global optimized face processor instance
face_processor = OptimizedFaceProcessor()
