"""Validation utilities for the face verification system."""

import os
import logging
from typing import Optional, List, Tuple
from pathlib import Path

from fastapi import UploadFile, HTTPException
from PIL import Image
import io

from src.config.settings import settings

logger = logging.getLogger(__name__)


class FileValidator:
    """File validation utilities."""
    
    @staticmethod
    def validate_image_file(file: UploadFile) -> Tuple[bool, Optional[str]]:
        """Validate uploaded image file."""
        try:
            # Check file size
            if hasattr(file, 'size') and file.size > settings.max_file_size:
                return False, f"File size exceeds maximum allowed size of {settings.max_file_size} bytes"
            
            # Check file extension
            if file.filename:
                file_ext = Path(file.filename).suffix.lower()
                allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
                if file_ext not in allowed_extensions:
                    return False, f"File extension {file_ext} not allowed. Allowed: {allowed_extensions}"
            
            # Check content type
            if file.content_type and not file.content_type.startswith('image/'):
                return False, f"Invalid content type: {file.content_type}"
            
            return True, None
            
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return False, f"File validation failed: {str(e)}"
    
    @staticmethod
    async def validate_image_content(image_bytes: bytes) -> Tuple[bool, str, Optional[Image.Image]]:
        """Validate image content by attempting to open it."""
        try:
            # Try to open as image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Validate image properties
            width, height = image.size
            
            if width < 50 or height < 50:
                return False, "Image dimensions too small (minimum 50x50)", None
            
            if width > 4000 or height > 4000:
                return False, "Image dimensions too large (maximum 4000x4000)", None
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return True, None, image
            
        except Exception as e:
            logger.error(f"Image content validation error: {e}")
            return False, f"Invalid image content: {str(e)}", None


class InputValidator:
    """Input validation utilities."""
    
    @staticmethod
    def validate_user_id(user_id: str) -> Tuple[bool, Optional[str]]:
        """Validate user ID format."""
        if not user_id or not user_id.strip():
            return False, "User ID cannot be empty"
        
        if len(user_id) > 100:
            return False, "User ID too long (maximum 100 characters)"
        
        # Check for invalid characters
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in invalid_chars:
            if char in user_id:
                return False, f"User ID contains invalid character: {char}"
        
        return True, None
    
    @staticmethod
    def validate_name(name: Optional[str]) -> Tuple[bool, Optional[str]]:
        """Validate user name."""
        if name is None:
            return True, None
        
        if len(name) > 200:
            return False, "Name too long (maximum 200 characters)"
        
        return True, None
    
    @staticmethod
    def validate_threshold(threshold: float) -> Tuple[bool, Optional[str]]:
        """Validate similarity threshold."""
        if not 0.0 <= threshold <= 1.0:
            return False, "Threshold must be between 0.0 and 1.0"
        
        return True, None


def validate_request_data(user_id: Optional[str] = None, 
                         name: Optional[str] = None,
                         threshold: Optional[float] = None) -> None:
    """Validate request data and raise HTTPException if invalid."""
    
    if user_id is not None:
        valid, error = InputValidator.validate_user_id(user_id)
        if not valid:
            raise HTTPException(status_code=400, detail=error)
    
    if name is not None:
        valid, error = InputValidator.validate_name(name)
        if not valid:
            raise HTTPException(status_code=400, detail=error)
    
    if threshold is not None:
        valid, error = InputValidator.validate_threshold(threshold)
        if not valid:
            raise HTTPException(status_code=400, detail=error)
