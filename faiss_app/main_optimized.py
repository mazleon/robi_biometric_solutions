"""Optimized FAISS GPU Microservice - Main FastAPI Application with Multiple Vector DB Support."""

import os
import time
import logging
from typing import List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from models import (
    VectorAddRequest, VectorBatchAddRequest, VectorSearchRequest,
    VectorSearchResponse, VectorRemoveRequest, IndexStatsResponse,
    HealthResponse, ErrorResponse
)
from alternative_vector_db_service import get_vector_service
from config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the configured vector database service
vector_service = get_vector_service()

# Create FastAPI app
app = FastAPI(
    title="Optimized Vector Database Microservice",
    description="High-performance vector similarity search supporting FAISS-GPU, ChromaDB, and Qdrant",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )

@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    db_type = os.getenv("VECTOR_DB_TYPE", "faiss").upper()
    logger.info(f"Optimized Vector Database Microservice starting up...")
    logger.info(f"Database Type: {db_type}")
    logger.info(f"Configuration: GPU Memory={config.GPU_MEMORY_GB}GB, FP16={config.USE_FP16}")
    logger.info(f"Embedding Dimension: {config.EMBEDDING_DIM}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Optimized Vector Database Microservice shutting down...")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        stats = vector_service.get_stats()
        db_type = stats.get("database_type", "FAISS")
        
        return HealthResponse(
            status="healthy",
            gpu_available=stats.get("gpu_enabled", False),
            faiss_version=f"{db_type} v2.0.0",
            total_vectors=stats.get("total_vectors", 0)
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )

@app.post("/vectors/add")
async def add_vector(request: VectorAddRequest):
    """Add a single vector to the index."""
    try:
        vector_id = vector_service.add_vector(
            user_id=request.user_id,
            embedding=request.embedding,
            metadata=request.metadata
        )
        
        return {
            "status": "success",
            "vector_id": vector_id,
            "user_id": request.user_id,
            "message": f"Vector added successfully with ID {vector_id}"
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to add vector: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add vector"
        )

@app.post("/vectors/batch-add")
async def batch_add_vectors(request: VectorBatchAddRequest):
    """Add multiple vectors to the index."""
    try:
        results = []
        failed = []
        
        # Check if the service supports batch operations
        if hasattr(vector_service, 'add_vector_batch'):
            # Use optimized batch operation
            try:
                vectors_data = [
                    {
                        'user_id': vec.user_id,
                        'embedding': vec.embedding,
                        'metadata': vec.metadata
                    }
                    for vec in request.vectors
                ]
                vector_ids = vector_service.add_vector_batch(vectors_data)
                
                for vec, vector_id in zip(request.vectors, vector_ids):
                    results.append({
                        "user_id": vec.user_id,
                        "vector_id": vector_id,
                        "status": "success"
                    })
                    
            except Exception as e:
                logger.error(f"Batch operation failed, falling back to individual adds: {e}")
                # Fallback to individual operations
                for vector_req in request.vectors:
                    try:
                        vector_id = vector_service.add_vector(
                            user_id=vector_req.user_id,
                            embedding=vector_req.embedding,
                            metadata=vector_req.metadata
                        )
                        results.append({
                            "user_id": vector_req.user_id,
                            "vector_id": vector_id,
                            "status": "success"
                        })
                    except Exception as e:
                        failed.append({
                            "user_id": vector_req.user_id,
                            "error": str(e)
                        })
        else:
            # Individual operations for services that don't support batch
            for vector_req in request.vectors:
                try:
                    vector_id = vector_service.add_vector(
                        user_id=vector_req.user_id,
                        embedding=vector_req.embedding,
                        metadata=vector_req.metadata
                    )
                    results.append({
                        "user_id": vector_req.user_id,
                        "vector_id": vector_id,
                        "status": "success"
                    })
                except Exception as e:
                    failed.append({
                        "user_id": vector_req.user_id,
                        "error": str(e)
                    })
        
        return {
            "status": "completed",
            "successful": len(results),
            "failed": len(failed),
            "results": results,
            "errors": failed
        }
        
    except Exception as e:
        logger.error(f"Batch add failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch add operation failed"
        )

@app.post("/vectors/search", response_model=VectorSearchResponse)
async def search_vectors(request: VectorSearchRequest):
    """Search for similar vectors."""
    try:
        start_time = time.time()
        
        results = vector_service.search_vectors(
            query_embedding=request.embedding,
            k=request.k,
            threshold=request.threshold
        )
        
        query_time_ms = (time.time() - start_time) * 1000
        stats = vector_service.get_stats()
        
        return VectorSearchResponse(
            results=results,
            query_time_ms=query_time_ms,
            total_vectors=stats.get("total_vectors", 0)
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search operation failed"
        )

@app.delete("/vectors/{user_id}")
async def remove_vector(user_id: str):
    """Remove a vector by user ID."""
    try:
        success = vector_service.remove_vector(user_id)
        
        if success:
            return {
                "status": "success",
                "user_id": user_id,
                "message": "Vector removed successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Vector not found for user {user_id}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove vector: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove vector"
        )

@app.get("/stats", response_model=IndexStatsResponse)
async def get_stats():
    """Get index statistics."""
    try:
        stats = vector_service.get_stats()
        
        return IndexStatsResponse(
            total_vectors=stats.get("total_vectors", 0),
            dimension=stats.get("dimension", config.EMBEDDING_DIM),
            index_type=stats.get("database_type", "Unknown"),
            gpu_enabled=stats.get("gpu_enabled", False),
            memory_usage_mb=stats.get("memory_usage_mb", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get statistics"
        )

@app.post("/admin/reset")
async def reset_index():
    """Reset the entire index (admin operation)."""
    try:
        success = vector_service.reset_index()
        
        if success:
            return {
                "status": "success",
                "message": "Index reset successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reset index"
            )
            
    except Exception as e:
        logger.error(f"Failed to reset index: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset index"
        )

@app.get("/")
async def root():
    """Root endpoint."""
    db_type = os.getenv("VECTOR_DB_TYPE", "faiss").upper()
    return {
        "service": "Optimized Vector Database Microservice",
        "version": "2.0.0",
        "database_type": db_type,
        "status": "running",
        "docs": "/docs"
    }

@app.get("/performance/benchmark")
async def performance_benchmark():
    """Run a quick performance benchmark."""
    try:
        import numpy as np
        
        # Generate test data
        test_embedding = np.random.rand(config.EMBEDDING_DIM).tolist()
        
        # Measure search time
        start_time = time.time()
        results = vector_service.search_vectors(test_embedding, k=5)
        search_time_ms = (time.time() - start_time) * 1000
        
        stats = vector_service.get_stats()
        
        return {
            "database_type": stats.get("database_type", "Unknown"),
            "total_vectors": stats.get("total_vectors", 0),
            "search_time_ms": round(search_time_ms, 2),
            "results_found": len(results),
            "gpu_enabled": stats.get("gpu_enabled", False)
        }
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Benchmark failed"
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(
        app,
        host=config.HOST,
        port=port,
        log_level=config.LOG_LEVEL.lower(),
        reload=False
    )
