"""Pydantic schemas for API request/response models."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class BaseResponse(BaseModel):
    """Base model for API responses, including processing time."""
    success: bool = Field(..., description="Indicates if the request was successful")
    message: Optional[str] = Field(None, description="A message providing details about the response")
    process_time_ms: Optional[float] = Field(None, description="Request processing time in milliseconds")


class EnrollRequest(BaseModel):
    """Request model for user enrollment."""
    user_id: str = Field(..., description="Unique user identifier", min_length=1, max_length=100)
    name: Optional[str] = Field(None, description="User display name", max_length=200)
    metadata: Optional[str] = Field(None, description="Additional user metadata")
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if not v.strip():
            raise ValueError('User ID cannot be empty or whitespace')
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in invalid_chars:
            if char in v:
                raise ValueError(f'User ID contains invalid character: {char}')
        return v.strip()


class EnrollResponse(BaseResponse):
    """Response model for user enrollment."""
    user_id: Optional[str] = Field(None, description="Enrolled user ID")
    face_count: Optional[int] = Field(None, description="Number of faces detected")
    vector_id: Optional[str] = Field(None, description="Qdrant vector point ID")


class VerifyResponse(BaseResponse):
    """Response model for face verification."""
    verified: bool = Field(False, description="Whether face was verified")
    user_id: Optional[str] = Field(None, description="Matched user ID")
    name: Optional[str] = Field(None, description="Matched user name")
    confidence: Optional[float] = Field(None, description="Verification confidence score")
    threshold: float = Field(..., description="Threshold used for verification")
    candidates: List[Dict[str, Any]] = Field(default_factory=list, description="All candidate matches")


class FaceInfo(BaseModel):
    """Face detection information."""
    index: int = Field(..., description="Face index in image")
    bbox: List[float] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")
    area: float = Field(..., description="Face area in pixels")
    confidence: float = Field(..., description="Detection confidence score")
    landmarks: Optional[List[List[float]]] = Field(None, description="Facial landmarks")


class DetectResponse(BaseResponse):
    """Response model for face detection."""
    face_count: int = Field(..., description="Number of faces detected")
    faces: List[FaceInfo] = Field(default_factory=list, description="Detected face information")


class UserInfo(BaseModel):
    """User information model."""
    user_id: str = Field(..., description="User ID")
    name: Optional[str] = Field(None, description="User name")
    metadata: Optional[str] = Field(None, description="User metadata")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class UsersListResponse(BaseResponse):
    """Response model for listing users."""
    users: List[UserInfo] = Field(default_factory=list, description="List of users")
    total_count: int = Field(..., description="Total number of users")


class DeleteUserResponse(BaseResponse):
    """Response model for user deletion."""
    user_id: Optional[str] = Field(None, description="Deleted user ID")


class SystemStatsResponse(BaseResponse):
    """Response model for system statistics."""
    database_stats: Dict[str, Any] = Field(default_factory=dict, description="Database statistics")
    vector_store_stats: Dict[str, Any] = Field(default_factory=dict, description="Vector store statistics")
    system_info: Dict[str, Any] = Field(default_factory=dict, description="System information")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="Application uptime in seconds")


class ResetResponse(BaseResponse):
    """Response model for database reset."""
    database_cleared: bool = Field(..., description="Whether database was cleared successfully")
    collection_cleared: bool = Field(..., description="Whether Qdrant collection was cleared successfully")
    stats_before: Dict[str, Any] = Field(default_factory=dict, description="Statistics before reset")
    stats_after: Dict[str, Any] = Field(default_factory=dict, description="Statistics after reset")


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
