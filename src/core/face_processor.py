"""Face processing module with InsightFace integration - optimized for GPU."""

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


class FaceProcessor:
    """Face processing class for detection, alignment, and embedding extraction."""
    
    def __init__(self):
        self.detector = None
        self.recognizer = None
        self.face_app = None
        self._model_warmed_up = False
        self._initialize_models()
        self._warmup_models()
    
    def _initialize_models(self):
        """Initialize InsightFace models for detection and recognition."""
        try:
            # Configure ONNX providers based on settings
            import onnxruntime as ort
            ort.set_default_logger_severity(3)  # Reduce ONNX logging
            
            # Configure ONNX Runtime providers with conservative GPU settings
            if settings.use_gpu and 'CUDAExecutionProvider' in settings.onnx_providers:
                provider_options = [{
                        'device_id': str(settings.gpu_device_id),
                        'arena_extend_strategy': 'kSameAsRequested',  # More conservative
                        'gpu_mem_limit': int(16 * 1024 * 1024 * 1024),  # 16GB limit (conservative)
                        'cudnn_conv_algo_search': 'HEURISTIC',  # Faster, less aggressive
                        'do_copy_in_default_stream': True,
                        'enable_cuda_graph': False,  # Disable CUDA graph capture
                        'tunable_op_enable': False,  # Disable tunable ops
                        'enable_skip_layer_norm_strict_mode': False,
                        'use_tf32': False  # Disable TF32 for compatibility
                    }]
                providers = list(zip(settings.onnx_providers, provider_options + [{}]))
            else:
                providers = settings.onnx_providers
            
            # Initialize face analysis with GPU/CPU configuration
            self.face_app = insightface.app.FaceAnalysis(
                name=settings.model_name,
                allowed_modules=['detection', 'recognition'],
                providers=providers
            )
            
            # Prepare with runtime configuration - use configured detection size
            self.face_app.prepare(
                ctx_id=settings.ctx_id,
                det_size=settings.detection_size
            )
            
            # Keep references for backward compatibility
            self.detector = self.face_app
            self.recognizer = self.face_app
            
            runtime_info = settings.get_runtime_info()
            logger.info(f"Face processing models initialized successfully with {runtime_info}")
            
            # Verify GPU utilization
            self._verify_gpu_usage()
            
        except Exception as e:
            logger.error(f"Failed to initialize face processing models: {e}")
            raise
    
    def _warmup_models(self):
        """Advanced GPU warmup for RTX 4090 optimization."""
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
        """Verify that models are actually using GPU."""
        try:
            import torch
            if torch.cuda.is_available() and settings.use_gpu:
                # Check GPU memory usage before and after a dummy inference
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated(0)
                
                # Run a small inference to check GPU usage
                dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                _ = self.face_app.get(dummy_image)
                
                final_memory = torch.cuda.memory_allocated(0)
                memory_used_mb = (final_memory - initial_memory) / (1024 * 1024)
                
                if memory_used_mb > 10:  # If more than 10MB used, likely on GPU
                    logger.info(f"✓ Models verified on GPU - Memory used: {memory_used_mb:.1f}MB")
                else:
                    logger.warning(f"⚠ Models may not be using GPU - Memory used: {memory_used_mb:.1f}MB")
                
                # Log GPU utilization stats
                perf_stats = gpu_optimizer.get_performance_stats()
                if perf_stats:
                    logger.info(f"GPU Stats: {perf_stats['device_name']}, "
                              f"Utilization: {perf_stats.get('memory_utilization', 0)*100:.1f}%")
            else:
                logger.info("GPU verification skipped - GPU not available or not enabled")
                
        except Exception as e:
            logger.warning(f"GPU verification failed: {e}")
    
    def image_bytes_to_pil(self, data: bytes) -> Image.Image:
        """Convert image bytes to PIL Image."""
        try:
            return Image.open(io.BytesIO(data)).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to convert bytes to PIL image: {e}")
            raise ValueError("Invalid image data")
    
    def validate_image(self, image: Image.Image) -> bool:
        """Validate image dimensions and format."""
        try:
            # Check image size
            width, height = image.size
            if width < 50 or height < 50:
                logger.warning(f"Image too small: {width}x{height}")
                return False
            
            if width > 4000 or height > 4000:
                logger.warning(f"Image too large: {width}x{height}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False
    
    
    def get_multiple_embeddings(self, image: Image.Image) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """Extract embeddings from all detected faces - optimized for speed."""
        try:
            # Fast validation - only check critical properties
            width, height = image.size
            if width < 50 or height < 50 or width > 4000 or height > 4000:
                logger.warning(f"Invalid image size: {width}x{height}")
                return []

            # GPU-optimized preprocessing and detection
            img_array = self._preprocess_image_fast(image)
            
            # Validate array shape before processing
            if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                logger.error(f"Invalid preprocessed image shape: {img_array.shape}")
                return []
            
            # Skip GPU optimizer calls if not critical for performance
            if settings.use_gpu:
                img_array = gpu_optimizer.optimize_memory_layout(img_array)
                gpu_optimizer.prefetch_to_gpu(img_array)
            
            # Use GPU-optimized detection with limited face count for speed
            faces = self.face_app.get(img_array, max_num=5)  # Reduced from 20 for faster processing

            if not faces:
                logger.debug("No faces detected in image")  # Changed to debug to reduce logging overhead
                return []

            embeddings_with_info = []
            for i, face in enumerate(faces):
                if face.embedding is None:
                    continue  # Skip logging for performance

                # Fast normalize embedding - use in-place operations
                embedding = face.embedding.astype('float32')
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding /= norm  # In-place division

                # Minimal face info for speed
                bbox = face.bbox
                face_info = {
                    'index': i,
                    'bbox': bbox.tolist(),
                    'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    'confidence': float(face.det_score)
                }
                embeddings_with_info.append((embedding, face_info))

            # Sort by face area (largest first) for consistent selection
            if len(embeddings_with_info) > 1:
                embeddings_with_info.sort(key=lambda x: x[1]['area'], reverse=True)
            
            logger.debug(f"Extracted {len(embeddings_with_info)} embeddings")  # Changed to debug
            return embeddings_with_info

        except Exception as e:
            logger.error(f"Failed to extract embeddings: {e}")
            return []
    
    def _preprocess_image_fast(self, image: Image.Image) -> np.ndarray:
        """Fast image preprocessing optimized for enrollment speed."""
        try:
            # Fast RGB conversion
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Aggressive resizing for speed - smaller images process faster
            width, height = image.size
            max_size = min(settings.max_image_size, 1024)  # Cap at 1024 for speed
            
            # Only resize if significantly larger (reduces processing time)
            if max(width, height) > max_size:
                ratio = max_size / max(width, height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                # Use faster resampling for enrollment (accuracy vs speed tradeoff)
                image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
            
            # Direct numpy conversion - fastest path
            img_array = np.ascontiguousarray(np.array(image, dtype=np.uint8))
            
            # Quick shape validation
            if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                if len(img_array.shape) == 2:  # Grayscale - fast conversion
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif len(img_array.shape) == 3 and img_array.shape[2] > 3:
                    img_array = img_array[:, :, :3]  # Take first 3 channels
                else:
                    raise ValueError(f"Unsupported image shape: {img_array.shape}")
            
            return img_array
            
        except Exception as e:
            logger.error(f"Fast preprocessing failed: {e}")
            # Simple fallback
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return np.ascontiguousarray(np.array(image, dtype=np.uint8))
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """GPU-optimized image preprocessing for RTX 4090 (legacy method)."""
        # For backward compatibility, delegate to fast method
        return self._preprocess_image_fast(image)
    
    def calculate_embedding_hash(self, embedding: np.ndarray) -> str:
        """Calculate hash of embedding for deduplication."""
        try:
            # Convert to bytes and hash
            embedding_bytes = embedding.tobytes()
            return hashlib.sha256(embedding_bytes).hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate embedding hash: {e}")
            return ""
    
    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Ensure embeddings are normalized
            emb1_norm = emb1 / np.linalg.norm(emb1)
            emb2_norm = emb2 / np.linalg.norm(emb2)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to compare embeddings: {e}")
            return 0.0


# Global face processor instance
face_processor = FaceProcessor()
