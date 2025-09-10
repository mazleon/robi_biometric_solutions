"""Data models for FAISS GPU service."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np


class VectorAddRequest(BaseModel):
    """Request model for adding vectors to FAISS index."""
    user_id: str = Field(..., description="User ID associated with the vector")
    embedding: List[float] = Field(..., description="Face embedding vector")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class VectorBatchAddRequest(BaseModel):
    """Request model for batch adding vectors to FAISS index."""
    vectors: List[VectorAddRequest] = Field(..., description="List of vectors to add")


class VectorSearchRequest(BaseModel):
    """Request model for searching similar vectors."""
    embedding: List[float] = Field(..., description="Query embedding vector")
    k: int = Field(default=5, ge=1, le=100, description="Number of similar vectors to return")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity threshold")


class VectorSearchResult(BaseModel):
    """Search result model."""
    user_id: str = Field(..., description="User ID")
    similarity: float = Field(..., description="Similarity score")
    faiss_id: int = Field(..., description="FAISS internal ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class VectorSearchResponse(BaseModel):
    """Response model for vector search."""
    results: List[VectorSearchResult] = Field(..., description="Search results")
    query_time_ms: float = Field(..., description="Query execution time in milliseconds")
    total_vectors: int = Field(..., description="Total vectors in index")


class VectorRemoveRequest(BaseModel):
    """Request model for removing vectors."""
    user_id: str = Field(..., description="User ID to remove")


class IndexStatsResponse(BaseModel):
    """Response model for index statistics."""
    total_vectors: int = Field(..., description="Total number of vectors")
    dimension: int = Field(..., description="Vector dimension")
    index_type: str = Field(..., description="FAISS index type")
    gpu_enabled: bool = Field(..., description="Whether GPU is enabled")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    gpu_available: bool = Field(..., description="GPU availability")
    faiss_version: str = Field(..., description="FAISS version")
    total_vectors: int = Field(..., description="Total vectors in index")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
