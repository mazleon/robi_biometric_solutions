"""Face processing module with InsightFace integration - optimized for GPU."""

import io
import hashlib
import logging
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from PIL import Image
from PIL import ImageOps, ImageEnhance
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
                        'gpu_mem_limit': int(22 * 1024 * 1024 * 1024),  # 16GB limit (conservative)
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
            
            # Initialize face analysis with optimized configuration
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
            
            # Configure detector for better recall
            self._configure_detector_for_accuracy()
            
            # Initialize multi-scale detection parameters for challenging images
            self.detection_scales = [0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0]  # Extended multi-scale detection
            self.min_face_size = 15  # Minimum detectable face size
            self.close_up_scales = [0.3, 0.4, 0.5]  # Special scales for close-up portraits
            
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
    
    def _configure_detector_for_accuracy(self):
        """Configure detector for maximum accuracy on challenging datasets."""
        try:
            # Access the detection model directly
            if hasattr(self.face_app, 'models') and 'detection' in self.face_app.models:
                detector = self.face_app.models['detection']
                
                # Configure detection parameters for better recall
                if hasattr(detector, 'nms_threshold'):
                    detector.nms_threshold = 0.4  # Balanced NMS
                if hasattr(detector, 'det_threshold'):
                    detector.det_threshold = getattr(settings, 'detection_threshold', 0.3)
                
                # Configure anchor settings if available (for RetinaFace/SCRFD)
                if hasattr(detector, '_anchor_cfg'):
                    # Ensure small face detection
                    detector._anchor_cfg['min_sizes'] = [[16, 32], [64, 128], [256, 512]]
                    detector._anchor_cfg['steps'] = [8, 16, 32]
                
                logger.info("Detector configured for maximum accuracy")
            
            # Also try to configure via det_model if available
            if hasattr(self.face_app, 'det_model'):
                det_model = self.face_app.det_model
                if hasattr(det_model, 'nms_threshold'):
                    det_model.nms_threshold = 0.4
                if hasattr(det_model, 'det_threshold'):
                    det_model.det_threshold = getattr(settings, 'detection_threshold', 0.3)
                    
        except Exception as e:
            logger.debug(f"Detector configuration failed: {e}")
    
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
            
            # Multi-scale detection for maximum recall
            faces = self._detect_faces_multiscale(img_array, image)
            
            if not faces:
                logger.debug("No faces detected after comprehensive multi-scale detection")
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
        """Optimized preprocessing for maximum detection accuracy."""
        try:
            # Handle EXIF orientation first
            image = ImageOps.exif_transpose(image)
            
            # Convert to RGB with proper color space handling
            if image.mode != 'RGB':
                if image.mode == 'RGBA':
                    # Handle transparency properly
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])
                    image = background
                else:
                    image = image.convert('RGB')
            
            # Intelligent resizing - preserve small faces
            width, height = image.size
            max_size = settings.max_image_size  # Use full configured size
            
            # Calculate minimum face size (assume 20px minimum face)
            min_face_size = 20
            min_dimension = min(width, height)
            
            # Only resize if image is significantly larger AND won't destroy small faces
            if max(width, height) > max_size and min_dimension > min_face_size * 4:
                # Use high-quality resampling for better face preservation
                ratio = max_size / max(width, height)
                new_width = max(int(width * ratio), min_face_size * 2)
                new_height = max(int(height * ratio), min_face_size * 2)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Enhance image quality for better detection
            if min_dimension < 400:  # Enhance small/low-quality images
                # Slight sharpening for small images
                image = ImageEnhance.Sharpness(image).enhance(1.1)
                # Slight contrast enhancement
                image = ImageEnhance.Contrast(image).enhance(1.05)
            
            # Convert to numpy with optimal memory layout
            img_array = np.ascontiguousarray(np.array(image, dtype=np.uint8))
            
            # Robust shape validation and correction
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack([img_array] * 3, axis=-1)
            elif len(img_array.shape) == 3:
                if img_array.shape[2] == 1:  # Single channel
                    img_array = np.repeat(img_array, 3, axis=2)
                elif img_array.shape[2] == 4:  # RGBA
                    img_array = img_array[:, :, :3]  # Drop alpha
                elif img_array.shape[2] != 3:
                    raise ValueError(f"Unsupported number of channels: {img_array.shape[2]}")
            else:
                raise ValueError(f"Unsupported image shape: {img_array.shape}")
            
            return img_array
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            # Robust fallback
            try:
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                return np.ascontiguousarray(np.array(image, dtype=np.uint8))
            except:
                # Last resort: create a dummy image
                return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """GPU-optimized image preprocessing for RTX 4090 (legacy method)."""
        # For backward compatibility, delegate to fast method
        return self._preprocess_image_fast(image)
    
    def _detect_faces_multiscale(self, img_array: np.ndarray, original_image: Image.Image) -> List[Any]:
        """Multi-scale face detection for maximum recall on challenging datasets."""
        all_faces = []
        max_faces = int(getattr(settings, 'max_faces_per_image', 5))
        
        try:
            # Primary detection at original scale
            faces = self.face_app.get(img_array, max_num=max_faces)
            if faces:
                all_faces.extend(faces)
                logger.debug(f"Primary scale detected {len(faces)} faces")
            
            # If no faces found, try comprehensive multi-scale detection
            if len(all_faces) == 0:
                # First try close-up portrait scales for images like the sample
                for scale in self.close_up_scales:
                    try:
                        w, h = original_image.size
                        new_w = max(int(w * scale), 100)
                        new_h = max(int(h * scale), 100)
                        
                        scaled_image = original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        scaled_array = self._preprocess_image_fast(scaled_image)
                        
                        # Use lower threshold for close-up detection
                        scale_faces = self._detect_with_threshold(scaled_array, 0.2)
                        
                        if scale_faces:
                            # Scale back the bounding boxes to original coordinates
                            for face in scale_faces:
                                if hasattr(face, 'bbox'):
                                    face.bbox = face.bbox / scale
                            
                            all_faces.extend(scale_faces)
                            logger.debug(f"Close-up scale {scale:.1f}x detected {len(scale_faces)} faces")
                            break
                    
                    except Exception as e:
                        logger.debug(f"Close-up detection at {scale:.1f}x failed: {e}")
                        continue
            
            # If still no faces, try standard multi-scale detection
            if len(all_faces) == 0:
                for scale in self.detection_scales:
                    if scale == 1.0:  # Skip original scale
                        continue
                    
                    try:
                        # Scale the image
                        w, h = original_image.size
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        
                        # Ensure reasonable size limits
                        max_dim = settings.max_image_size * 1.5  # Allow more overflow for detection
                        if max(new_w, new_h) > max_dim:
                            continue
                        if min(new_w, new_h) < 30:  # Reduced minimum size
                            continue
                        
                        scaled_image = original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        scaled_array = self._preprocess_image_fast(scaled_image)
                        
                        # Detect faces at this scale with adaptive threshold
                        threshold = 0.25 if scale < 1.0 else 0.3
                        scale_faces = self._detect_with_threshold(scaled_array, threshold)
                        
                        if scale_faces:
                            # Scale back the bounding boxes to original coordinates
                            for face in scale_faces:
                                if hasattr(face, 'bbox'):
                                    face.bbox = face.bbox / scale
                            
                            all_faces.extend(scale_faces)
                            logger.debug(f"Scale {scale:.1f}x detected {len(scale_faces)} additional faces")
                            
                            # If we found enough faces, stop
                            if len(all_faces) >= max_faces:
                                break
                    
                    except Exception as e:
                        logger.debug(f"Multi-scale detection at {scale:.1f}x failed: {e}")
                        continue
            
            # Fallback strategies if still no faces
            if not all_faces:
                all_faces = self._fallback_detection_strategies(img_array, original_image)
            
            # Remove duplicates and sort by confidence
            if len(all_faces) > 1:
                all_faces = self._deduplicate_faces(all_faces)
            
            # Limit to max faces
            return all_faces[:max_faces]
            
        except Exception as e:
            logger.error(f"Multi-scale detection failed: {e}")
            return []
    
    def _detect_with_threshold(self, img_array: np.ndarray, threshold: float) -> List[Any]:
        """Detect faces with a specific threshold."""
        try:
            # Try to set threshold if possible
            original_thresh = None
            detector = None
            
            if hasattr(self.face_app, 'models') and 'detection' in self.face_app.models:
                detector = self.face_app.models['detection']
            elif hasattr(self.face_app, 'det_model'):
                detector = self.face_app.det_model
            
            if detector and hasattr(detector, 'det_threshold'):
                original_thresh = detector.det_threshold
                detector.det_threshold = threshold
            
            # Perform detection
            faces = self.face_app.get(img_array, max_num=5)
            
            # Restore original threshold
            if detector and original_thresh is not None:
                detector.det_threshold = original_thresh
            
            return faces
            
        except Exception as e:
            logger.debug(f"Threshold detection failed: {e}")
            return []
    
    def _fallback_detection_strategies(self, img_array: np.ndarray, original_image: Image.Image) -> List[Any]:
        """Advanced fallback strategies for difficult images like close-up portraits."""
        faces = []
        
        try:
            # Strategy 1: Aggressive threshold reduction for close-up portraits
            for thresh in [0.15, 0.1, 0.08, 0.05]:
                faces = self._detect_with_threshold(img_array, thresh)
                if faces:
                    logger.debug(f"Aggressive threshold {thresh} succeeded")
                    break
            
            # Strategy 2: Enhanced preprocessing for challenging lighting
            if not faces:
                try:
                    # Multiple enhancement approaches
                    enhancement_configs = [
                        {'contrast': 1.5, 'brightness': 1.2, 'sharpness': 1.3},
                        {'contrast': 0.8, 'brightness': 1.3, 'sharpness': 1.4},
                        {'contrast': 1.2, 'brightness': 0.9, 'sharpness': 1.5},
                    ]
                    
                    for config in enhancement_configs:
                        enhanced = ImageEnhance.Contrast(original_image).enhance(config['contrast'])
                        enhanced = ImageEnhance.Brightness(enhanced).enhance(config['brightness'])
                        enhanced = ImageEnhance.Sharpness(enhanced).enhance(config['sharpness'])
                        
                        enhanced_array = self._preprocess_image_fast(enhanced)
                        faces = self._detect_with_threshold(enhanced_array, 0.15)
                        
                        if faces:
                            logger.debug(f"Enhancement config {config} succeeded")
                            break
                        
                except Exception as e:
                    logger.debug(f"Enhancement fallback failed: {e}")
            
            # Strategy 3: Histogram equalization with multiple approaches
            if not faces:
                try:
                    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                    import cv2
                    img_cv = np.array(original_image)
                    
                    # Convert to LAB color space
                    lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    lab[:,:,0] = clahe.apply(lab[:,:,0])
                    enhanced_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                    
                    faces = self._detect_with_threshold(enhanced_cv, 0.15)
                    
                    if faces:
                        logger.debug("CLAHE enhancement succeeded")
                    
                except Exception as e:
                    logger.debug(f"CLAHE enhancement failed: {e}")
                    
                    # Fallback to simple histogram equalization
                    try:
                        gray = original_image.convert('L')
                        equalized = ImageOps.equalize(gray)
                        equalized_rgb = Image.merge('RGB', [equalized] * 3)
                        
                        eq_array = self._preprocess_image_fast(equalized_rgb)
                        faces = self._detect_with_threshold(eq_array, 0.12)
                        
                        if faces:
                            logger.debug("Simple histogram equalization succeeded")
                            
                    except Exception as e:
                        logger.debug(f"Histogram equalization failed: {e}")
            
            # Strategy 4: Edge enhancement for poorly defined faces
            if not faces:
                try:
                    from PIL import ImageFilter
                    
                    # Apply edge enhancement
                    edge_enhanced = original_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
                    # Blend with original
                    blended = Image.blend(original_image, edge_enhanced, 0.3)
                    
                    blended_array = self._preprocess_image_fast(blended)
                    faces = self._detect_with_threshold(blended_array, 0.1)
                    
                    if faces:
                        logger.debug("Edge enhancement succeeded")
                        
                except Exception as e:
                    logger.debug(f"Edge enhancement failed: {e}")
            
        except Exception as e:
            logger.error(f"Fallback strategies failed: {e}")
        
        return faces
    
    def _deduplicate_faces(self, faces: List[Any]) -> List[Any]:
        """Remove duplicate face detections based on IoU overlap."""
        if len(faces) <= 1:
            return faces
        
        try:
            # Calculate IoU between all pairs and remove duplicates
            unique_faces = []
            
            for face in faces:
                is_duplicate = False
                face_bbox = getattr(face, 'bbox', None)
                
                if face_bbox is None:
                    continue
                
                for unique_face in unique_faces:
                    unique_bbox = getattr(unique_face, 'bbox', None)
                    if unique_bbox is None:
                        continue
                    
                    # Calculate IoU
                    iou = self._calculate_bbox_iou(face_bbox, unique_bbox)
                    if iou > 0.5:  # 50% overlap threshold
                        # Keep the one with higher confidence
                        face_conf = getattr(face, 'det_score', 0)
                        unique_conf = getattr(unique_face, 'det_score', 0)
                        
                        if face_conf > unique_conf:
                            # Replace the unique face with current face
                            idx = unique_faces.index(unique_face)
                            unique_faces[idx] = face
                        
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_faces.append(face)
            
            # Sort by confidence (highest first)
            unique_faces.sort(key=lambda x: getattr(x, 'det_score', 0), reverse=True)
            return unique_faces
            
        except Exception as e:
            logger.debug(f"Face deduplication failed: {e}")
            return faces
    
    def _calculate_bbox_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate Intersection over Union (IoU) for two bounding boxes."""
        try:
            # bbox format: [x1, y1, x2, y2]
            x1_inter = max(bbox1[0], bbox2[0])
            y1_inter = max(bbox1[1], bbox2[1])
            x2_inter = min(bbox1[2], bbox2[2])
            y2_inter = min(bbox1[3], bbox2[3])
            
            # Calculate intersection area
            if x2_inter <= x1_inter or y2_inter <= y1_inter:
                return 0.0
            
            intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            
            # Calculate union area
            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
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
