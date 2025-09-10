"""FAISS GPU Microservice - Main FastAPI Application."""

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
from faiss_service import faiss_service
from config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="FAISS GPU Microservice",
    description="High-performance vector similarity search using FAISS with GPU acceleration",
    version="1.0.0",
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
    logger.info("FAISS GPU Microservice starting up...")
    logger.info(f"Configuration: GPU Memory={config.GPU_MEMORY_GB}GB, FP16={config.USE_FP16}")
    logger.info(f"Index Type: {config.INDEX_TYPE}, Dimension: {config.EMBEDDING_DIM}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("FAISS GPU Microservice shutting down...")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        stats = faiss_service.get_stats()
        
        return HealthResponse(
            status="healthy",
            gpu_available=stats.get("gpu_enabled", False),
            faiss_version="1.8.0",  # Update based on actual version
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
        faiss_id = faiss_service.add_vector(
            user_id=request.user_id,
            embedding=request.embedding,
            metadata=request.metadata
        )
        
        return {
            "status": "success",
            "faiss_id": faiss_id,
            "user_id": request.user_id,
            "message": f"Vector added successfully with ID {faiss_id}"
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
        
        for vector_req in request.vectors:
            try:
                faiss_id = faiss_service.add_vector(
                    user_id=vector_req.user_id,
                    embedding=vector_req.embedding,
                    metadata=vector_req.metadata
                )
                results.append({
                    "user_id": vector_req.user_id,
                    "faiss_id": faiss_id,
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
        
        results = faiss_service.search_vectors(
            query_embedding=request.embedding,
            k=request.k,
            threshold=request.threshold
        )
        
        query_time_ms = (time.time() - start_time) * 1000
        stats = faiss_service.get_stats()
        
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
        success = faiss_service.remove_vector(user_id)
        
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
        stats = faiss_service.get_stats()
        
        return IndexStatsResponse(
            total_vectors=stats.get("total_vectors", 0),
            dimension=stats.get("dimension", config.EMBEDDING_DIM),
            index_type=stats.get("index_type", config.INDEX_TYPE),
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
        success = faiss_service.reset_index()
        
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

@app.post("/admin/migrate")
async def migrate_legacy_data():
    """Manually trigger legacy data migration."""
    try:
        from data_migration import run_migration
        
        # Run migration from default data directory
        legacy_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        migration_result = run_migration(legacy_data_dir)
        
        if migration_result['success']:
            return {
                "status": "success",
                "message": "Data migration completed successfully",
                "details": {
                    "vectors_migrated": migration_result.get('total_vectors', 0),
                    "users_migrated": migration_result.get('total_users', 0),
                    "faiss_migrated": migration_result.get('faiss_migrated', False),
                    "metadata_migrated": migration_result.get('metadata_migrated', False)
                }
            }
        else:
            return {
                "status": "error",
                "message": "Data migration failed",
                "errors": migration_result.get('errors', [])
            }
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Migration failed: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "FAISS GPU Microservice",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level=config.LOG_LEVEL.lower(),
        reload=False
    )
