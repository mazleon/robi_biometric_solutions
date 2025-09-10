"""Optimized API endpoints for the face verification system."""

import logging
import time
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from src.api.schemas import (
    EnrollResponse, VerifyResponse, DetectResponse,
    UsersListResponse, DeleteUserResponse, SystemStatsResponse,
    HealthResponse, FaceInfo, UserInfo, ResetResponse
)
from src.core.face_processor import face_processor
from src.core.hybrid_vector_store import hybrid_vector_store
# Database operations handled by hybrid vector store (Qdrant + FAISS)
from src.utils.validators import FileValidator, validate_request_data
from src.utils.timing import TimingTracker
from src.utils.performance_monitor import perf_monitor, monitor_performance
from src.utils.async_optimizer import optimize_async_endpoint, parallel_executor
from src.utils.memory_optimizer import memory_efficient, gpu_memory_efficient
from src.config.settings import settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Application start time for uptime calculation
app_start_time = time.time()


@router.post("/enroll", response_model=EnrollResponse)
@monitor_performance("enroll")
@optimize_async_endpoint(cache_ttl=0, max_concurrency=8)
@gpu_memory_efficient
async def enroll_user(
    user_id: str = Form(...),
    name: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    file: UploadFile = File(...)
):
    """Enroll a user with their face image - fully optimized for RTX 4090."""
    timing_tracker = TimingTracker()
    timing_tracker.start("total_enrollment")
    
    try:
        # Fast concurrent validation
        validate_request_data(user_id=user_id, name=name)
        timing_tracker.start("file_validation")
        file_valid, file_error = FileValidator.validate_image_file(file)
        if not file_valid:
            raise HTTPException(status_code=400, detail=file_error)
        timing_tracker.end("file_validation")

        # Read file content once and validate concurrently
        timing_tracker.start("file_processing")
        image_bytes = await file.read()
        content_valid, content_error, image = await FileValidator.validate_image_content(image_bytes)
        if not content_valid:
            raise HTTPException(status_code=400, detail=content_error)
        timing_tracker.end("file_processing")

        # Process image to get faces and embeddings
        timing_tracker.start("face_processing")
        embeddings_with_info = face_processor.get_multiple_embeddings(image)
        if not embeddings_with_info:
            timing_tracker.end("total_enrollment")
            return EnrollResponse(
                success=False,
                message="No faces detected or embeddings could not be extracted",
                face_count=0,
                process_time_ms=timing_tracker.get_timing("total_enrollment") * 1000 if timing_tracker.get_timing("total_enrollment") else 0
            )

        # Use the embedding from the largest face
        embedding, face_info = embeddings_with_info[0]
        timing_tracker.end("face_processing")

        # Pre-calculate embedding hash for parallel operations
        timing_tracker.start("hash_calculation")
        embedding_hash = face_processor.calculate_embedding_hash(embedding)
        timing_tracker.end("hash_calculation")

        # Store user data in Qdrant metadata
        timing_tracker.start("qdrant_operations")
        # Parse metadata if provided as JSON string
        parsed_metadata = {}
        if metadata:
            try:
                import json
                parsed_metadata = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                parsed_metadata = {"raw_metadata": metadata}
        
        user_metadata = {"name": name, "timestamp": time.time(), **parsed_metadata}
        timing_tracker.end("qdrant_operations")

        # Store embedding in hybrid vector store with performance tracking
        timing_tracker.start("hybrid_enrollment")
        hybrid_id = await hybrid_vector_store.add_embedding(user_id, embedding, user_metadata)
        timing_tracker.end("hybrid_enrollment")

        # Metadata stored in Qdrant vector - no separate database needed
        timing_tracker.start("metadata_finalization")
        # All metadata is stored with the vector in Qdrant
        timing_tracker.end("metadata_finalization")

        timing_tracker.end("total_enrollment")
        timing_tracker.log_summary("Face Enrollment", "total_enrollment")
        
        logger.info(f"Successfully enrolled user {user_id} with embedding hybrid ID {hybrid_id}")

        return EnrollResponse(
            success=True,
            user_id=user_id,
            message=f"User enrolled successfully with {len(embeddings_with_info)} faces detected",
            face_count=len(embeddings_with_info),
            embedding_id=hybrid_id,
            process_time_ms=timing_tracker.get_timing("total_enrollment") or (time.time() - start_time) * 1000
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enrollment failed for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")


@router.post("/verify", response_model=VerifyResponse)
async def verify_face(
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(None)
):
    """Verify a face against enrolled users."""
    start_time = time.time()
    timing_tracker = TimingTracker()
    timing_tracker.start("total_verification")
    
    try:
        # Use default threshold if not provided
        verification_threshold = threshold if threshold is not None else settings.cosine_threshold
        validate_request_data(threshold=verification_threshold)
        
        # Validate file
        file_valid, file_error = FileValidator.validate_image_file(file)
        if not file_valid:
            raise HTTPException(status_code=400, detail=file_error)
        
        # Read and validate image content
        timing_tracker.start("file_processing")
        image_bytes = await file.read()
        content_valid, content_error, image = await FileValidator.validate_image_content(image_bytes)
        if not content_valid:
            raise HTTPException(status_code=400, detail=content_error)
        timing_tracker.end("file_processing")
        
        # Process image to get faces and embeddings
        timing_tracker.start("face_detection_recognition")
        embeddings_with_info = face_processor.get_multiple_embeddings(image)
        timing_tracker.end("face_detection_recognition")
        if not embeddings_with_info:
            return VerifyResponse(
                success=True,
                verified=False,
                threshold=verification_threshold,
                message="No faces detected or embeddings could not be extracted",
                process_time_ms=timing_tracker.get_timing("total_verification") or (time.time() - start_time) * 1000
            )

        # Use the embedding from the largest face
        embedding, _ = embeddings_with_info[0]
        
        # Search for similar embeddings in hybrid vector store with performance tracking
        timing_tracker.start("hybrid_search")
        search_results = await hybrid_vector_store.search_similar(embedding, k=5, strategy="hybrid")
        timing_tracker.end("hybrid_search")
        
        # Determine if verification passed
        verified = False
        matched_user = None
        confidence = 0.0
        
        if search_results:
            best_match = search_results[0]
            confidence = best_match.similarity
            
            # Log detailed similarity scores for debugging
            logger.info(f"Verification attempt - Top candidates:")
            for i, result in enumerate(search_results[:3]):  # Log top 3 matches
                logger.info(f"  {i+1}. User: {result.user_id}, Similarity: {result.similarity:.4f}, Source: {result.source}")
            
            if confidence >= verification_threshold:
                verified = True
                matched_user = {
                    'user_id': best_match.user_id,
                    'name': best_match.metadata.get('name'),
                    'similarity': best_match.similarity
                }
                logger.info(f"[PASS] Face VERIFIED for user '{matched_user['user_id']}' with confidence {confidence:.4f} (threshold: {verification_threshold:.4f}) via {best_match.source}")
            else:
                logger.info(f"[FAIL] Face NOT VERIFIED. Best match: '{best_match.user_id}' with confidence {confidence:.4f} < threshold {verification_threshold:.4f} via {best_match.source}")
        else:
            logger.info("[FAIL] No similar faces found in database")
        
        # Verification logging handled by Qdrant service
        logger.info(f"Verification attempt: user_id={matched_user['user_id'] if matched_user else None}, verified={verified}, confidence={confidence:.3f}, threshold={verification_threshold:.3f}")
        
        timing_tracker.end("total_verification")
        
        # Log comprehensive timing information with correct total operation
        timing_tracker.log_summary("Face Verification", "total_verification")
        
        logger.info(f"Verification completed: verified={verified}, confidence={confidence:.3f}")
        
        return VerifyResponse(
            success=True,
            verified=verified,
            user_id=matched_user['user_id'] if matched_user else None,
            name=matched_user['name'] if matched_user else None,
            confidence=confidence,
            threshold=verification_threshold,
            candidates=[
                {
                    'user_id': result.user_id,
                    'name': result.metadata.get('name'),
                    'similarity': result.similarity,
                    'source': result.source
                }
                for result in search_results
            ],
            message="Face verified successfully" if verified else "Face not recognized",
            process_time_ms=timing_tracker.get_timing("total_verification") or (time.time() - start_time) * 1000
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@router.post("/enroll", response_model=EnrollResponse)
async def enroll_user(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    request: Request = None
):
    """Enroll a new user with face embedding using hybrid vector store."""
    start_time = time.time()
    
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        if file.size > settings.max_file_size:
            raise HTTPException(status_code=400, detail=f"File size exceeds {settings.max_file_size} bytes")
        
        # Read and process image
        image_data = await file.read()
        
        # Extract face embedding
        result = face_processor.extract_embedding(image_data)
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
        
        embedding = result['embedding']
        face_info = result.get('face_info', {})
        
        # Get hybrid store and ensure it's initialized
        hybrid_store = request.app.state.hybrid_store
        
        # Ensure collection exists before adding embedding
        try:
            await hybrid_store.qdrant.initialize()
        except Exception as init_error:
            logger.warning(f"Hybrid store initialization during enrollment: {init_error}")
        
        success = await hybrid_store.add_embedding(
            user_id=user_id,
            embedding=embedding,
            metadata={
                'enrolled_at': time.time(),
                'face_count': face_info.get('face_count', 1),
                'confidence': face_info.get('confidence', 0.0),
                'image_size': len(image_data)
            }
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store embedding")
        
        # Record performance metrics
        processing_time = (time.time() - start_time) * 1000
        perf_monitor.record_request('enrollment', processing_time / 1000, 0)
        
        return EnrollResponse(
            success=True,
            message=f"User {user_id} enrolled successfully",
            user_id=user_id,
            process_time_ms=processing_time,
            face_count=face_info.get('face_count', 1)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enrollment failed for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {e}")


@router.post("/detect", response_model=DetectResponse)
async def detect_faces(file: UploadFile = File(...)):
    """Detect faces in an image without verification."""
    start_time = time.time()
    try:
        # Validate file
        file_valid, file_error = FileValidator.validate_image_file(file)
        if not file_valid:
            raise HTTPException(status_code=400, detail=file_error)

        # Read and validate image content
        image_bytes = await file.read()
        content_valid, content_error, image = await FileValidator.validate_image_content(image_bytes)
        if not content_valid:
            raise HTTPException(status_code=400, detail=content_error)
        
        # Process image to get face information
        embeddings_with_info = face_processor.get_multiple_embeddings(image)
        
        # Convert to response format
        faces = [
            FaceInfo(
                index=info['index'],
                bbox=info['bbox'],
                area=info['area'],
                confidence=info['confidence']
            )
            for _, info in embeddings_with_info
        ]
        
        logger.info(f"Detected {len(faces)} faces in image")
        
        return DetectResponse(
            success=True,
            face_count=len(faces),
            faces=faces,
            message=f"Detected {len(faces)} faces",
            process_time_ms=(time.time() - start_time) * 1000
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Face detection failed: {str(e)}")


@router.get("/users", response_model=UsersListResponse)
async def list_users():
    """List all enrolled users."""
    start_time = time.time()
    try:
        users = db_manager.get_all_users()
        
        user_info_list = [
            UserInfo(
                user_id=user.user_id,
                name=user.name,
                metadata=user.metadata,
                created_at=user.created_at,
                updated_at=user.updated_at
            )
            for user in users
        ]
        
        return UsersListResponse(
            success=True,
            users=user_info_list,
            total_count=len(user_info_list),
            process_time_ms=(time.time() - start_time) * 1000
        )
        
    except Exception as e:
        logger.error(f"Failed to list users: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list users: {str(e)}")


@router.delete("/users/{user_id}", response_model=DeleteUserResponse)
async def delete_user(user_id: str):
    """Delete a user and their face data."""
    start_time = time.time()
    try:
        validate_request_data(user_id=user_id)
        
        # Check if user exists in hybrid vector store and delete
        user_exists = await hybrid_vector_store.user_exists(user_id)
        if not user_exists:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Delete from hybrid vector store (both Qdrant and FAISS)
        vector_deleted = await hybrid_vector_store.delete_user_embeddings(user_id)
        if not vector_deleted:
            logger.warning(f"Failed to delete vectors for user {user_id} from hybrid vector store")
        
        return DeleteUserResponse(
            success=True,
            user_id=user_id,
            message="User deleted successfully from hybrid vector store (Qdrant + FAISS)",
            process_time_ms=(time.time() - start_time) * 1000
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete user: {str(e)}")


@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats():
    """Get system statistics."""
    start_time = time.time()
    try:
        # Get comprehensive hybrid vector store stats with performance metrics
        hybrid_stats = await hybrid_vector_store.get_stats()
        
        db_stats = {
            'total_users': hybrid_stats.total_vectors,
            'qdrant_vectors': hybrid_stats.qdrant_vectors,
            'faiss_vectors': hybrid_stats.faiss_vectors,
            'cache_hit_rate': hybrid_stats.cache_hit_rate,
            'hybrid_service': f"{settings.qdrant_url} + FAISS GPU"
        }
        
        # System info with runtime configuration
        runtime_info = settings.get_runtime_info()
        system_info = {
            'embedding_dimension': settings.embedding_dim,
            'cosine_threshold': settings.cosine_threshold,
            'detection_size': settings.detection_size,
            'max_file_size': settings.max_file_size,
            'allowed_extensions': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'],
            'runtime_mode': 'GPU' if runtime_info['use_gpu'] else 'CPU',
            'model_name': runtime_info['model_name'],
            'onnx_providers': runtime_info['onnx_providers'],
            'qdrant_enabled': runtime_info['qdrant_enabled'],
            'gpu_device_id': runtime_info['gpu_device_id'],
            'batch_size': runtime_info['batch_size'],
            'max_image_size': runtime_info['max_image_size']
        }
        
        
        return SystemStatsResponse(
            success=True,
            database_stats=db_stats,
            vector_store_stats={
                'total_vectors': hybrid_stats.total_vectors,
                'qdrant_vectors': hybrid_stats.qdrant_vectors,
                'faiss_vectors': hybrid_stats.faiss_vectors,
                'cache_hit_rate': hybrid_stats.cache_hit_rate,
                'avg_search_time_ms': hybrid_stats.avg_search_time_ms,
                'qdrant_health': hybrid_stats.qdrant_health,
                'faiss_health': hybrid_stats.faiss_health,
                'sync_status': hybrid_stats.sync_status,
                'memory_usage_mb': hybrid_stats.memory_usage_mb
            },
            system_info=system_info,
            process_time_ms=(time.time() - start_time) * 1000
        )
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system stats: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        uptime = time.time() - app_start_time
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="0.1.0",
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/users/{user_id}/image")
async def get_user_image(user_id: str):
    """Get the latest image for a user."""
    try:
        # Validate user_id
        validate_request_data(user_id=user_id)
        
        # Check if user exists in hybrid vector store
        user_exists = await hybrid_vector_store.user_exists(user_id)
        if not user_exists:
            raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
        
        # Get image data from hybrid vector store metadata
        # Note: This would need to be implemented in hybrid_vector_store
        image_data = None  # Placeholder - implement get_user_image in hybrid store
        if not image_data:
            raise HTTPException(status_code=404, detail=f"No image found for user '{user_id}'")
            
        # Determine content type based on image data
        content_type = "image/jpeg"  # Default
        if image_data.startswith(b'\x89PNG'):
            content_type = "image/png"
        elif image_data.startswith(b'\xff\xd8'):
            content_type = "image/jpeg"
        elif image_data.startswith(b'RIFF') and b'WEBP' in image_data[:12]:
            content_type = "image/webp"
            
        logger.info(f"Retrieved image for user {user_id}, size: {len(image_data)} bytes")
        return Response(content=image_data, media_type=content_type)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get image for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve image: {str(e)}")


@router.get("/performance/qdrant")
async def get_qdrant_performance():
    """Get detailed Qdrant performance metrics and operation statistics."""
    try:
        # Get comprehensive performance stats from hybrid vector store
        hybrid_stats = await hybrid_vector_store.get_stats()
        performance_metrics = hybrid_vector_store.get_performance_metrics()
        
        # Get performance monitor stats for hybrid operations
        perf_stats = perf_monitor.get_stats()
        hybrid_endpoints = {k: v for k, v in perf_stats.get('endpoints', {}).items() 
                           if any(term in k.lower() for term in ['qdrant', 'faiss', 'hybrid'])}
        
        return {
            "success": True,
            "hybrid_service": {
                "qdrant_url": settings.qdrant_url,
                "faiss_enabled": settings.faiss_enabled,
                "qdrant_healthy": hybrid_stats.qdrant_health,
                "faiss_healthy": hybrid_stats.faiss_health,
                "sync_status": hybrid_stats.sync_status,
                "cache_hit_rate": hybrid_stats.cache_hit_rate
            },
            "vector_database": {
                'total_vectors': hybrid_stats.total_vectors,
                'qdrant_vectors': hybrid_stats.qdrant_vectors,
                'faiss_vectors': hybrid_stats.faiss_vectors,
                'avg_search_time_ms': hybrid_stats.avg_search_time_ms,
                'memory_usage_mb': hybrid_stats.memory_usage_mb
            },
            "operation_performance": hybrid_endpoints,
            "performance_metrics": performance_metrics,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get Qdrant performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


@router.post("/reset", response_model=ResetResponse)
async def reset_system():
    """Reset the entire system by clearing database and Qdrant collection."""
    start_time = time.time()
    try:
        # Get statistics before reset
        stats_before = await hybrid_vector_store.get_stats()
        
        logger.warning("System reset initiated - this will clear all data")
        
        # Database operations handled by hybrid vector store
        logger.info("All data operations handled by hybrid vector store (Qdrant + FAISS)")
        
        # Reset hybrid vector store (both Qdrant and FAISS)
        index_reset_success = await hybrid_vector_store.reset_all()
        if not index_reset_success:
            raise HTTPException(status_code=500, detail="Failed to reset hybrid vector store")
        
        # Get statistics after reset
        stats_after = await hybrid_vector_store.get_stats()
        
        logger.warning("System reset completed successfully")
        
        return ResetResponse(
            success=True,
            message="System reset completed successfully - all data cleared",
            database_cleared=True,
            index_cleared=index_reset_success,
            stats_before=stats_before,
            stats_after=stats_after,
            process_time_ms=(time.time() - start_time) * 1000
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"System reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"System reset failed: {str(e)}")


