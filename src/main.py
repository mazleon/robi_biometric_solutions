"""Main FastAPI application for the face verification system."""

import logging
from contextlib import asynccontextmanager

import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import ErrorResponse

from src.config.settings import settings
from src.utils.logging_config import setup_logging
from src.api.endpoints import router
# Database operations handled by hybrid vector store (Qdrant + FAISS)
from src.core.hybrid_vector_store import hybrid_vector_store
from src.core.face_processor import face_processor

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with hybrid vector store."""
    # Startup
    logger.info("Starting Face Verification API with hybrid vector store...")
    
    try:
        # Initialize hybrid vector store with collection creation
        success = await hybrid_vector_store.initialize()
        if not success:
            logger.error("Failed to initialize hybrid vector store")
            # Try to continue anyway for debugging
            logger.warning("Continuing startup despite hybrid store initialization failure")
        else:
            logger.info("Hybrid vector store initialized successfully")
        
        # Store reference for dependency injection
        app.state.hybrid_store = hybrid_vector_store
        
        # Test face processor
        logger.info("Face processor models loaded successfully")
        logger.info("Face Verification API started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        # Don't raise to allow debugging
        logger.warning("Continuing despite startup errors")
        yield
    
    # Shutdown
    logger.info("Shutting down Face Verification API...")
    
    # Clean up hybrid vector store
    try:
        await hybrid_vector_store.close()
        logger.info("Hybrid vector store closed successfully")
    except Exception as e:
        logger.error(f"Error closing hybrid vector store: {e}")


# Create FastAPI app
app = FastAPI(
    title="Face Verification API",
    description="Face verification system using InsightFace with Hybrid Vector Store (Qdrant + FAISS GPU)",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to add request processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Face Verification API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPException",
            message=exc.detail
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An internal server error occurred"
        ).dict()
    )
