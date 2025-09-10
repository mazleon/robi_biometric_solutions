"""FAISS GPU Vector Store Service."""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import faiss
from config import config
from models import VectorSearchResult

logger = logging.getLogger(__name__)


class FaissGPUService:
    """FAISS GPU service for vector operations."""
    
    def __init__(self):
        self.index = None
        self.gpu_resources = None
        self.gpu_index = None
        self.metadata: Dict[int, Dict[str, Any]] = {}
        self.user_id_to_faiss_id: Dict[str, int] = {}
        self.faiss_id_to_user_id: Dict[int, str] = {}
        self.next_faiss_id = 0
        
        # Check for existing data migration
        self._check_and_migrate_legacy_data()
        
        self._initialize_gpu()
        self._initialize_index()
        self._load_metadata()
    
    def _initialize_gpu(self):
        """Initialize GPU resources for FAISS."""
        try:
            if not hasattr(faiss, 'StandardGpuResources'):
                logger.warning("FAISS GPU support not available, falling back to CPU")
                self.gpu_resources = None
                return
            
            self.gpu_resources = faiss.StandardGpuResources()
            self.gpu_resources.setDefaultNullStreamAllDevices()
            
            # Configure GPU memory
            if config.GPU_MEMORY_GB > 0:
                memory_bytes = config.GPU_MEMORY_GB * 1024 * 1024 * 1024
                self.gpu_resources.setTempMemory(memory_bytes)
            
            logger.info(f"GPU resources initialized for device {config.GPU_DEVICE_ID}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize GPU resources: {e}, falling back to CPU")
            self.gpu_resources = None
    
    def _initialize_index(self):
        """Initialize or load FAISS index."""
        try:
            if os.path.exists(config.INDEX_PATH):
                # Load existing index
                self.index = faiss.read_index(config.INDEX_PATH)
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            else:
                # Create new index based on configuration
                if config.INDEX_TYPE == "IndexFlatIP":
                    # Inner product (cosine similarity)
                    base_index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
                elif config.INDEX_TYPE == "IndexFlatL2":
                    # L2 distance
                    base_index = faiss.IndexFlatL2(config.EMBEDDING_DIM)
                else:
                    raise ValueError(f"Unsupported index type: {config.INDEX_TYPE}")
                
                # Wrap with IndexIDMap for custom IDs
                self.index = faiss.IndexIDMap(base_index)
                self._save_index()
                logger.info(f"Created new FAISS index: {config.INDEX_TYPE}")
            
            # Move to GPU
            self._move_to_gpu()
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise
    
    def _move_to_gpu(self):
        """Move FAISS index to GPU."""
        try:
            # Check if GPU resources are available
            if not hasattr(faiss, 'StandardGpuResources') or not self.gpu_resources:
                logger.info("GPU resources not available, using CPU-only FAISS")
                self.gpu_index = None
                return
            
            # Move index to GPU without additional config for compatibility
            self.gpu_index = faiss.index_cpu_to_gpu(
                self.gpu_resources,
                config.GPU_DEVICE_ID,
                self.index
            )
            
            logger.info(f"FAISS index moved to GPU {config.GPU_DEVICE_ID}")
            
        except Exception as e:
            logger.warning(f"Failed to move index to GPU, using CPU: {e}")
            self.gpu_index = None
    
    def _get_active_index(self):
        """Get the active index (GPU preferred, CPU fallback)."""
        return self.gpu_index if self.gpu_index is not None else self.index
    
    def _save_index(self):
        """Save FAISS index to disk."""
        try:
            os.makedirs(os.path.dirname(config.INDEX_PATH), exist_ok=True)
            
            # Always save CPU version for persistence
            if self.gpu_index is not None:
                cpu_index = faiss.index_gpu_to_cpu(self.gpu_index)
                faiss.write_index(cpu_index, config.INDEX_PATH)
            else:
                faiss.write_index(self.index, config.INDEX_PATH)
            
            logger.debug(f"FAISS index saved to {config.INDEX_PATH}")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise
    
    def _load_metadata(self):
        """Load metadata from disk."""
        try:
            if os.path.exists(config.METADATA_PATH):
                with open(config.METADATA_PATH, 'r') as f:
                    data = json.load(f)
                    self.metadata = {int(k): v for k, v in data.get('metadata', {}).items()}
                    self.user_id_to_faiss_id = data.get('user_id_to_faiss_id', {})
                    self.faiss_id_to_user_id = {int(k): v for k, v in data.get('faiss_id_to_user_id', {}).items()}
                    self.next_faiss_id = data.get('next_faiss_id', 0)
                
                logger.info(f"Loaded metadata for {len(self.metadata)} vectors")
            
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
    
    def _save_metadata(self):
        """Save metadata to disk."""
        try:
            os.makedirs(os.path.dirname(config.METADATA_PATH), exist_ok=True)
            
            data = {
                'metadata': {str(k): v for k, v in self.metadata.items()},
                'user_id_to_faiss_id': self.user_id_to_faiss_id,
                'faiss_id_to_user_id': {str(k): v for k, v in self.faiss_id_to_user_id.items()},
                'next_faiss_id': self.next_faiss_id
            }
            
            with open(config.METADATA_PATH, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Metadata saved to disk")
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def add_vector(self, user_id: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add a vector to the index."""
        try:
            # Convert to numpy array and normalize
            vector = np.array(embedding, dtype=np.float32)
            if vector.shape[0] != config.EMBEDDING_DIM:
                raise ValueError(f"Invalid embedding dimension: {vector.shape[0]}, expected: {config.EMBEDDING_DIM}")
            
            # Normalize for cosine similarity
            vector = vector / np.linalg.norm(vector)
            
            # Generate FAISS ID
            faiss_id = self.next_faiss_id
            self.next_faiss_id += 1
            
            # Add to index
            active_index = self._get_active_index()
            active_index.add_with_ids(
                np.expand_dims(vector, axis=0),
                np.array([faiss_id], dtype=np.int64)
            )
            
            # Store metadata
            self.metadata[faiss_id] = metadata or {}
            self.user_id_to_faiss_id[user_id] = faiss_id
            self.faiss_id_to_user_id[faiss_id] = user_id
            
            # Save to disk
            self._save_index()
            self._save_metadata()
            
            logger.info(f"Added vector for user {user_id} with FAISS ID {faiss_id}")
            return faiss_id
            
        except Exception as e:
            logger.error(f"Failed to add vector for user {user_id}: {e}")
            raise
    
    def search_vectors(self, query_embedding: List[float], k: int = 5, threshold: Optional[float] = None) -> List[VectorSearchResult]:
        """Search for similar vectors."""
        try:
            start_time = time.time()
            
            active_index = self._get_active_index()
            if active_index.ntotal == 0:
                return []
            
            # Convert and normalize query
            query = np.array(query_embedding, dtype=np.float32)
            query = query / np.linalg.norm(query)
            
            # Search
            similarities, faiss_ids = active_index.search(
                np.expand_dims(query, axis=0), 
                min(k, active_index.ntotal)
            )
            
            # Process results
            results = []
            for similarity, faiss_id in zip(similarities[0], faiss_ids[0]):
                if faiss_id == -1:  # Invalid ID
                    continue
                
                faiss_id = int(faiss_id)
                similarity = float(similarity)
                
                # Apply threshold if specified
                if threshold is not None and similarity < threshold:
                    continue
                
                # Get user info
                user_id = self.faiss_id_to_user_id.get(faiss_id)
                if user_id:
                    results.append(VectorSearchResult(
                        user_id=user_id,
                        similarity=similarity,
                        faiss_id=faiss_id,
                        metadata=self.metadata.get(faiss_id, {})
                    ))
            
            query_time = (time.time() - start_time) * 1000  # Convert to ms
            logger.info(f"Search completed in {query_time:.2f}ms, found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            raise
    
    def remove_vector(self, user_id: str) -> bool:
        """Remove a vector from the index."""
        try:
            faiss_id = self.user_id_to_faiss_id.get(user_id)
            if faiss_id is None:
                logger.warning(f"No vector found for user {user_id}")
                return False
            
            # FAISS doesn't support direct removal, so we mark as removed
            # In a production system, you'd implement periodic index rebuilding
            if faiss_id in self.metadata:
                del self.metadata[faiss_id]
            if user_id in self.user_id_to_faiss_id:
                del self.user_id_to_faiss_id[user_id]
            if faiss_id in self.faiss_id_to_user_id:
                del self.faiss_id_to_user_id[faiss_id]
            
            self._save_metadata()
            
            logger.info(f"Marked vector for user {user_id} as removed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove vector for user {user_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            active_index = self._get_active_index()
            
            return {
                'total_vectors': active_index.ntotal,
                'dimension': config.EMBEDDING_DIM,
                'index_type': config.INDEX_TYPE,
                'gpu_enabled': self.gpu_index is not None,
                'memory_usage_mb': self._get_memory_usage(),
                'metadata_count': len(self.metadata)
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def _get_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        try:
            if os.path.exists(config.INDEX_PATH):
                return os.path.getsize(config.INDEX_PATH) / (1024 * 1024)
            return 0.0
        except:
            return 0.0
    
    def reset_index(self) -> bool:
        """Reset the entire index."""
        try:
            # Create new empty index
            if config.INDEX_TYPE == "IndexFlatIP":
                base_index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
            else:
                base_index = faiss.IndexFlatL2(config.EMBEDDING_DIM)
            
            self.index = faiss.IndexIDMap(base_index)
            self._move_to_gpu()
            
            # Clear metadata
            self.metadata.clear()
            self.user_id_to_faiss_id.clear()
            self.faiss_id_to_user_id.clear()
            self.next_faiss_id = 0
            
            # Save
            self._save_index()
            self._save_metadata()
            
            logger.info(f"FAISS index reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset index: {e}")
            return False
    
    def _check_and_migrate_legacy_data(self):
        """Check for legacy data and migrate if needed."""
        try:
            # Check if we already have migrated data
            if os.path.exists(config.INDEX_PATH) and os.path.exists(config.METADATA_PATH):
                logger.info("Migrated data already exists, skipping migration")
                return
            
            # Check for legacy data in mounted directory or parent directory
            legacy_data_dir = "/legacy_data" if os.path.exists("/legacy_data") else os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
            legacy_db_path = os.path.join(legacy_data_dir, "users.db")
            legacy_index_path = os.path.join(legacy_data_dir, "faiss.index")
            
            if os.path.exists(legacy_db_path) or os.path.exists(legacy_index_path):
                logger.info("Found legacy data, starting migration...")
                
                from data_migration import run_migration
                migration_result = run_migration(legacy_data_dir)
                
                if migration_result['success']:
                    logger.info(f"Data migration completed successfully:")
                    logger.info(f"  - Vectors migrated: {migration_result.get('total_vectors', 0)}")
                    logger.info(f"  - Users migrated: {migration_result.get('total_users', 0)}")
                    logger.info(f"  - FAISS index: {'✓' if migration_result.get('faiss_migrated') else '✗'}")
                    logger.info(f"  - Metadata: {'✓' if migration_result.get('metadata_migrated') else '✗'}")
                else:
                    logger.error(f"Data migration failed: {migration_result.get('errors', [])}")
            else:
                logger.info("No legacy data found, starting with empty index")
                
        except Exception as e:
            logger.error(f"Legacy data migration check failed: {e}")


# Global service instance
faiss_service = FaissGPUService()
