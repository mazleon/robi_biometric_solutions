"""Consolidated utility functions for face recognition backend."""

import hashlib
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)


def calculate_hash(data: bytes) -> str:
    """Calculate SHA-256 hash of data."""
    return hashlib.sha256(data).hexdigest()


def calculate_embedding_hash(embedding: np.ndarray) -> str:
    """Calculate hash of face embedding for deduplication."""
    try:
        return hashlib.sha256(embedding.tobytes()).hexdigest()
    except Exception as e:
        logger.error(f"Failed to calculate embedding hash: {e}")
        return ""


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize embedding vector using L2 norm."""
    try:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    except Exception as e:
        logger.error(f"Failed to normalize embedding: {e}")
        return embedding


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    try:
        emb1_norm = normalize_embedding(emb1)
        emb2_norm = normalize_embedding(emb2)
        return float(np.dot(emb1_norm, emb2_norm))
    except Exception as e:
        logger.error(f"Failed to calculate cosine similarity: {e}")
        return 0.0


def validate_image_dimensions(image: Image.Image, 
                            min_size: int = 50, 
                            max_size: int = 4000) -> Tuple[bool, str]:
    """Validate image dimensions."""
    width, height = image.size
    
    if width < min_size or height < min_size:
        return False, f"Image too small: {width}x{height}. Minimum: {min_size}x{min_size}"
    
    if width > max_size or height > max_size:
        return False, f"Image too large: {width}x{height}. Maximum: {max_size}x{max_size}"
    
    return True, "Valid dimensions"


def optimize_image_for_processing(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """Optimize image for face processing."""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large
        width, height = image.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        
        return image
    except Exception as e:
        logger.error(f"Failed to optimize image: {e}")
        return image


def format_face_info(face_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format face detection information."""
    return {
        'bbox': face_data.get('bbox', []),
        'confidence': float(face_data.get('confidence', 0.0)),
        'area': float(face_data.get('area', 0.0)),
        'index': int(face_data.get('index', 0))
    }


def format_response_timing(timing_data: Dict[str, float]) -> Dict[str, Any]:
    """Format timing information for API responses."""
    return {
        'total_time_ms': round(timing_data.get('total', 0) * 1000, 2),
        'face_detection_ms': round(timing_data.get('face_detection', 0) * 1000, 2),
        'embedding_extraction_ms': round(timing_data.get('embedding_extraction', 0) * 1000, 2),
        'vector_store_ms': round(timing_data.get('vector_store', 0) * 1000, 2)
    }


def create_error_response(error_code: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create standardized error response."""
    response = {
        'success': False,
        'error': {
            'code': error_code,
            'message': message,
            'timestamp': time.time()
        }
    }
    
    if details:
        response['error']['details'] = details
    
    return response


def create_success_response(data: Dict[str, Any], timing: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Create standardized success response."""
    response = {
        'success': True,
        'data': data,
        'timestamp': time.time()
    }
    
    if timing:
        response['timing'] = format_response_timing(timing)
    
    return response


def validate_user_id(user_id: str) -> Tuple[bool, str]:
    """Validate user ID format."""
    if not user_id or not isinstance(user_id, str):
        return False, "User ID must be a non-empty string"
    
    if len(user_id) < 3:
        return False, "User ID must be at least 3 characters long"
    
    if len(user_id) > 50:
        return False, "User ID must be no more than 50 characters long"
    
    # Check for valid characters (alphanumeric, underscore, hyphen)
    if not user_id.replace('_', '').replace('-', '').isalnum():
        return False, "User ID can only contain alphanumeric characters, underscores, and hyphens"
    
    return True, "Valid user ID"


def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int_conversion(value: Any, default: int = 0) -> int:
    """Safely convert value to integer."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def truncate_string(text: str, max_length: int = 100) -> str:
    """Truncate string to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def format_bytes(bytes_value: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024
    return f"{bytes_value:.1f} TB"


def get_file_extension(filename: str) -> str:
    """Get file extension in lowercase."""
    return filename.lower().split('.')[-1] if '.' in filename else ''


def is_valid_image_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """Check if file has valid image extension."""
    extension = f".{get_file_extension(filename)}"
    return extension in [ext.lower() for ext in allowed_extensions]


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.elapsed_time()
        logger.debug(f"{self.name} completed in {duration:.3f}s")
    
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time


def batch_process(items: List[Any], batch_size: int = 10):
    """Generator to process items in batches."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry function on failure."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")
            
            raise last_exception
        
        return wrapper
    return decorator
