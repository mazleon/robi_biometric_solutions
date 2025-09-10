"""Optimized FAISS GPU Vector Store Service with Performance Enhancements."""

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


class OptimizedFaissGPUService:
    """Optimized FAISS GPU service for enhanced vector operations performance."""
    
    def __init__(self):
        self.index = None
        self.gpu_resources = None
        self.gpu_index = None
        self.metadata: Dict[int, Dict[str, Any]] = {}
        self.user_id_to_faiss_id: Dict[str, int] = {}
        self.faiss_id_to_user_id: Dict[int, str] = {}
        self.next_faiss_id = 0
        
        # Performance optimization variables
        self._batch_buffer = []
        self._batch_size = config.BATCH_SIZE
        self._last_save_time = time.time()
        self._save_interval = 30  # Save every 30 seconds
        
        # Check for existing data migration
        self._check_and_migrate_legacy_data()
        
        self._initialize_gpu()
        self._initialize_optimized_index()
        self._load_metadata()
    
    def _initialize_gpu(self):
        """Initialize GPU resources with optimized settings."""
        try:
            if not hasattr(faiss, 'StandardGpuResources'):
                logger.warning("FAISS GPU support not available, falling back to CPU")
                self.gpu_resources = None
                return
            
            self.gpu_resources = faiss.StandardGpuResources()
            
            # Optimize GPU memory allocation
            if config.GPU_MEMORY_GB > 0:
                memory_bytes = config.GPU_MEMORY_GB * 1024 * 1024 * 1024
                self.gpu_resources.setTempMemory(memory_bytes)
            
            # Set optimal stream configuration for better GPU utilization
            self.gpu_resources.setDefaultNullStreamAllDevices()
            
            logger.info(f"Optimized GPU resources initialized for device {config.GPU_DEVICE_ID}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize GPU resources: {e}, falling back to CPU")
            self.gpu_resources = None
    
    def _initialize_optimized_index(self):
        """Initialize FAISS index with performance optimizations."""
        try:
            if os.path.exists(config.INDEX_PATH):
                # Load existing index
                self.index = faiss.read_index(config.INDEX_PATH)
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            else:
                # Create optimized index based on expected dataset size
                if config.INDEX_TYPE == "IndexFlatIP":
                    base_index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
                elif config.INDEX_TYPE == "IndexFlatL2":
                    base_index = faiss.IndexFlatL2(config.EMBEDDING_DIM)
                elif config.INDEX_TYPE == "IndexIVFFlat":
                    # Use IVF for larger datasets (>100K vectors)
                    quantizer = faiss.IndexFlatIP(config.EMBEDDING_DIM)
                    nlist = min(4096, max(100, int(np.sqrt(config.MAX_VECTORS))))
                    base_index = faiss.IndexIVFFlat(quantizer, config.EMBEDDING_DIM, nlist)
                elif config.INDEX_TYPE == "IndexHNSW":
                    # HNSW for very fast search with memory trade-off
                    base_index = faiss.IndexHNSWFlat(config.EMBEDDING_DIM, 32)
                    base_index.hnsw.efConstruction = 200
                    base_index.hnsw.efSearch = 50
                else:
                    base_index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
                
                # Wrap with IndexIDMap for custom IDs
                self.index = faiss.IndexIDMap(base_index)
                self._save_index()
                logger.info(f"Created optimized FAISS index: {config.INDEX_TYPE}")
            
            # Move to GPU with optimized configuration
            self._move_to_gpu_optimized()
            
        except Exception as e:
            logger.error(f"Failed to initialize optimized FAISS index: {e}")
            raise
    
    def _move_to_gpu_optimized(self):
        """Move FAISS index to GPU with optimized configuration."""
        try:
            if not hasattr(faiss, 'StandardGpuResources') or not self.gpu_resources:
                logger.info("GPU resources not available, using CPU-only FAISS")
                self.gpu_index = None
                return
            
            # Configure GPU index with optimizations
            gpu_config = faiss.GpuIndexFlatConfig()
            gpu_config.device = config.GPU_DEVICE_ID
            gpu_config.useFloat16 = config.USE_FP16  # Use FP16 for memory efficiency
            gpu_config.storeTransposed = True  # Better memory access patterns
            
            # Move index to GPU with configuration
            if isinstance(self.index.index, faiss.IndexFlat):
                self.gpu_index = faiss.index_cpu_to_gpu(
                    self.gpu_resources,
                    config.GPU_DEVICE_ID,
                    self.index,
                    gpu_config
                )
            else:
                # For other index types, use standard GPU transfer
                self.gpu_index = faiss.index_cpu_to_gpu(
                    self.gpu_resources,
                    config.GPU_DEVICE_ID,
                    self.index
                )
            
            logger.info(f"FAISS index moved to GPU {config.GPU_DEVICE_ID} with optimizations")
            
        except Exception as e:
            logger.warning(f"Failed to move index to GPU with optimizations, using CPU: {e}")
            self.gpu_index = None
    
    def _get_active_index(self):
        """Get the active index (GPU preferred, CPU fallback)."""
        return self.gpu_index if self.gpu_index is not None else self.index
    
    def add_vector_batch(self, vectors_data: List[Dict]) -> List[int]:
        """Add multiple vectors in an optimized batch operation."""
        try:
            if not vectors_data:
                return []
            
            # Prepare batch data
            embeddings = []
            faiss_ids = []
            metadata_batch = {}
            user_mappings = {}
            
            for vector_data in vectors_data:
                user_id = vector_data['user_id']
                embedding = vector_data['embedding']
                metadata = vector_data.get('metadata', {})
                
                # Convert and normalize embedding
                vector = np.array(embedding, dtype=np.float32)
                if vector.shape[0] != config.EMBEDDING_DIM:
                    raise ValueError(f"Invalid embedding dimension: {vector.shape[0]}")
                
                vector = vector / np.linalg.norm(vector)
                embeddings.append(vector)
                
                # Generate FAISS ID
                faiss_id = self.next_faiss_id
                self.next_faiss_id += 1
                faiss_ids.append(faiss_id)
                
                # Store metadata and mappings
                metadata_batch[faiss_id] = metadata
                user_mappings[user_id] = faiss_id
            
            # Batch add to index
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss_ids_array = np.array(faiss_ids, dtype=np.int64)
            
            active_index = self._get_active_index()
            active_index.add_with_ids(embeddings_array, faiss_ids_array)
            
            # Update metadata and mappings
            self.metadata.update(metadata_batch)
            self.user_id_to_faiss_id.update(user_mappings)
            for faiss_id, user_id in zip(faiss_ids, [v['user_id'] for v in vectors_data]):
                self.faiss_id_to_user_id[faiss_id] = user_id
            
            # Conditional save (avoid frequent I/O)
            current_time = time.time()
            if current_time - self._last_save_time > self._save_interval:
                self._save_index()
                self._save_metadata()
                self._last_save_time = current_time
            
            logger.info(f"Batch added {len(faiss_ids)} vectors")
            return faiss_ids
            
        except Exception as e:
            logger.error(f"Failed to batch add vectors: {e}")
            raise
    
    def search_vectors_optimized(self, query_embedding: List[float], k: int = 5, 
                                threshold: Optional[float] = None, 
                                use_gpu_search: bool = True) -> List[VectorSearchResult]:
        """Optimized vector search with performance enhancements."""
        try:
            start_time = time.time()
            
            active_index = self._get_active_index()
            if active_index.ntotal == 0:
                return []
            
            # Convert and normalize query with optimized operations
            query = np.array(query_embedding, dtype=np.float32)
            query = query / np.linalg.norm(query)
            
            # Optimize search parameters based on index type
            search_k = min(k, active_index.ntotal)
            
            # Configure search parameters for different index types
            if hasattr(active_index, 'nprobe'):
                # For IVF indices, optimize nprobe
                active_index.nprobe = min(64, max(1, int(np.sqrt(active_index.nlist))))
            
            if hasattr(active_index, 'efSearch'):
                # For HNSW indices, optimize efSearch
                active_index.efSearch = max(search_k, 50)
            
            # Perform optimized search
            similarities, faiss_ids = active_index.search(
                np.expand_dims(query, axis=0), 
                search_k
            )
            
            # Process results with vectorized operations
            results = []
            valid_mask = faiss_ids[0] != -1
            
            for i, (similarity, faiss_id) in enumerate(zip(similarities[0], faiss_ids[0])):
                if not valid_mask[i]:
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
            
            query_time = (time.time() - start_time) * 1000
            logger.info(f"Optimized search completed in {query_time:.2f}ms, found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            raise
    
    def _save_index(self):
        """Save FAISS index to disk with optimizations."""
        try:
            os.makedirs(os.path.dirname(config.INDEX_PATH), exist_ok=True)
            
            # Always save CPU version for persistence
            if self.gpu_index is not None:
                cpu_index = faiss.index_gpu_to_cpu(self.gpu_index)
                faiss.write_index(cpu_index, config.INDEX_PATH)
            else:
                faiss.write_index(self.index, config.INDEX_PATH)
            
            logger.debug(f"Optimized FAISS index saved to {config.INDEX_PATH}")
            
        except Exception as e:
            logger.error(f"Failed to save optimized FAISS index: {e}")
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
            
            logger.debug("Optimized metadata saved to disk")
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def add_vector(self, user_id: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add a single vector (wrapper for batch operation)."""
        vector_data = {
            'user_id': user_id,
            'embedding': embedding,
            'metadata': metadata or {}
        }
        result = self.add_vector_batch([vector_data])
        return result[0] if result else None
    
    def search_vectors(self, query_embedding: List[float], k: int = 5, threshold: Optional[float] = None) -> List[VectorSearchResult]:
        """Search for similar vectors (wrapper for optimized search)."""
        return self.search_vectors_optimized(query_embedding, k, threshold)
    
    def remove_vector(self, user_id: str) -> bool:
        """Remove a vector from the index."""
        try:
            faiss_id = self.user_id_to_faiss_id.get(user_id)
            if faiss_id is None:
                logger.warning(f"No vector found for user {user_id}")
                return False
            
            # FAISS doesn't support direct removal, so we mark as removed
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
                'metadata_count': len(self.metadata),
                'optimization_enabled': True,
                'fp16_enabled': config.USE_FP16
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
            elif config.INDEX_TYPE == "IndexIVFFlat":
                quantizer = faiss.IndexFlatIP(config.EMBEDDING_DIM)
                nlist = min(4096, max(100, int(np.sqrt(config.MAX_VECTORS))))
                base_index = faiss.IndexIVFFlat(quantizer, config.EMBEDDING_DIM, nlist)
            elif config.INDEX_TYPE == "IndexHNSW":
                base_index = faiss.IndexHNSWFlat(config.EMBEDDING_DIM, 32)
                base_index.hnsw.efConstruction = 200
                base_index.hnsw.efSearch = 50
            else:
                base_index = faiss.IndexFlatL2(config.EMBEDDING_DIM)
            
            self.index = faiss.IndexIDMap(base_index)
            self._move_to_gpu_optimized()
            
            # Clear metadata
            self.metadata.clear()
            self.user_id_to_faiss_id.clear()
            self.faiss_id_to_user_id.clear()
            self.next_faiss_id = 0
            
            # Save
            self._save_index()
            self._save_metadata()
            
            logger.info(f"Optimized FAISS index reset successfully")
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
                logger.info("No legacy data found, starting with empty optimized index")
                
        except Exception as e:
            logger.error(f"Legacy data migration check failed: {e}")


# Global optimized service instance
optimized_faiss_service = OptimizedFaissGPUService()
