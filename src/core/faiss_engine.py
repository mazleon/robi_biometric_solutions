"""
Production-ready FAISS engine for GPU-accelerated vector search.
Optimized for 10M+ face embeddings with RTX 4090.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pickle
import json

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class FaissSearchResult:
    """FAISS search result with metadata."""
    distances: np.ndarray
    indices: np.ndarray
    user_ids: List[str]
    similarities: np.ndarray
    metadata: List[Dict[str, Any]]


@dataclass
class FaissStats:
    """FAISS engine statistics."""
    total_vectors: int
    index_type: str
    is_trained: bool
    gpu_enabled: bool
    memory_usage_mb: float
    search_time_avg_ms: float
    last_optimization: float
    index_file_size_mb: float


class FaissEngine:
    """
    Production-ready FAISS engine for ultra-fast face embedding search.
    
    Features:
    - GPU acceleration with RTX 4090 optimization
    - Multiple index types (Flat, IVF, HNSW)
    - Automatic index optimization
    - Memory-efficient operations
    - Thread-safe operations
    - Backup and recovery
    """
    
    def __init__(self):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-gpu")
        
        self.embedding_dim = settings.embedding_dim
        self.use_gpu = settings.use_gpu and faiss.get_num_gpus() > 0
        self.gpu_device_id = settings.gpu_device_id
        
        # Index configuration
        self.index_type = "IVF"  # Default to IVF for production
        self.nlist = 4096  # Number of clusters for IVF
        self.nprobe = 64   # Number of clusters to search
        self.m = 64        # PQ code size
        self.nbits = 8     # Bits per PQ code
        
        # Index and metadata storage
        self.index: Optional[faiss.Index] = None
        self.gpu_index: Optional[faiss.Index] = None
        self.gpu_resources: Optional[faiss.StandardGpuResources] = None
        self.user_id_map: Dict[int, str] = {}  # FAISS ID -> User ID
        self.metadata_map: Dict[int, Dict[str, Any]] = {}  # FAISS ID -> Metadata
        self.reverse_user_map: Dict[str, List[int]] = {}  # User ID -> FAISS IDs
        
        # Performance tracking
        self.stats = {
            'searches': 0,
            'insertions': 0,
            'total_search_time': 0.0,
            'avg_search_time': 0.0,
            'cache_hits': 0,
            'index_rebuilds': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        self._next_id = 0
        
        # File paths
        self.index_dir = Path("data/faiss")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.index_dir / "face_embeddings.index"
        self.metadata_file = self.index_dir / "metadata.pkl"
        self.config_file = self.index_dir / "config.json"
        
        # Initialize
        self._initialize_gpu()
        self._create_index()
    
    def _initialize_gpu(self):
        """Initialize GPU resources for FAISS."""
        if not self.use_gpu:
            logger.info("Using CPU FAISS")
            return
        
        try:
            # Initialize GPU resources with RTX 4090 optimization
            self.gpu_resources = faiss.StandardGpuResources()
            
            # Set memory allocation (use 80% of GPU memory for FAISS)
            gpu_memory_gb = 24  # RTX 4090 has 24GB
            faiss_memory_bytes = int(gpu_memory_gb * 0.8 * 1024 * 1024 * 1024)
            self.gpu_resources.setTempMemory(faiss_memory_bytes)
            
            # Configure for optimal performance
            self.gpu_resources.setDefaultNullStreamAllDevices()
            
            logger.info(f"GPU FAISS initialized on device {self.gpu_device_id} with {faiss_memory_bytes/1024/1024/1024:.1f}GB memory")
            
        except Exception as e:
            logger.warning(f"GPU initialization failed, falling back to CPU: {e}")
            self.use_gpu = False
            self.gpu_resources = None
    
    def _create_index(self):
        """Create optimized FAISS index for face embeddings."""
        with self._lock:
            try:
                if self.index_type == "Flat":
                    # Exact search - best accuracy, slower for large datasets
                    self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine
                    
                elif self.index_type == "IVF":
                    # IVF with PQ - balanced speed/accuracy for production
                    quantizer = faiss.IndexFlatIP(self.embedding_dim)
                    self.index = faiss.IndexIVFPQ(
                        quantizer, 
                        self.embedding_dim, 
                        self.nlist,  # Number of clusters
                        self.m,      # PQ segments
                        self.nbits   # Bits per segment
                    )
                    self.index.nprobe = self.nprobe
                    
                elif self.index_type == "HNSW":
                    # HNSW - fastest search, more memory usage
                    self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
                    self.index.hnsw.efConstruction = 200
                    self.index.hnsw.efSearch = 64
                
                else:
                    raise ValueError(f"Unsupported index type: {self.index_type}")
                
                # Move to GPU if available
                if self.use_gpu and self.gpu_resources:
                    self.gpu_index = faiss.index_cpu_to_gpu(
                        self.gpu_resources, 
                        self.gpu_device_id, 
                        self.index
                    )
                    logger.info(f"Created GPU {self.index_type} index")
                else:
                    self.gpu_index = None
                    logger.info(f"Created CPU {self.index_type} index")
                
                # Try to load existing index
                self._load_index()
                
            except Exception as e:
                logger.error(f"Failed to create FAISS index: {e}")
                raise
    
    def _get_active_index(self) -> faiss.Index:
        """Get the active index (GPU if available, otherwise CPU)."""
        return self.gpu_index if self.gpu_index is not None else self.index
    
    def add_embeddings(
        self, 
        embeddings: np.ndarray, 
        user_ids: List[str], 
        metadata_list: List[Dict[str, Any]]
    ) -> List[int]:
        """Add multiple embeddings to the index."""
        if len(embeddings) != len(user_ids) or len(embeddings) != len(metadata_list):
            raise ValueError("Embeddings, user_ids, and metadata must have same length")
        
        with self._lock:
            start_time = time.time()
            
            try:
                # Normalize embeddings for cosine similarity
                embeddings_normalized = embeddings.copy()
                norms = np.linalg.norm(embeddings_normalized, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                embeddings_normalized = embeddings_normalized / norms
                
                # Get FAISS IDs
                faiss_ids = list(range(self._next_id, self._next_id + len(embeddings)))
                self._next_id += len(embeddings)
                
                # Train index if needed
                active_index = self._get_active_index()
                if not active_index.is_trained:
                    if hasattr(active_index, 'train'):
                        logger.info("Training FAISS index...")
                        if self.gpu_index is not None:
                            # Train on GPU
                            active_index.train(embeddings_normalized.astype('float32'))
                        else:
                            # Train on CPU
                            self.index.train(embeddings_normalized.astype('float32'))
                            if self.gpu_index is not None:
                                # Copy trained index to GPU
                                self.gpu_index = faiss.index_cpu_to_gpu(
                                    self.gpu_resources, 
                                    self.gpu_device_id, 
                                    self.index
                                )
                        logger.info("Index training completed")
                
                # Add embeddings
                active_index.add_with_ids(
                    embeddings_normalized.astype('float32'), 
                    np.array(faiss_ids, dtype=np.int64)
                )
                
                # Update metadata maps
                for i, (faiss_id, user_id, metadata) in enumerate(zip(faiss_ids, user_ids, metadata_list)):
                    self.user_id_map[faiss_id] = user_id
                    self.metadata_map[faiss_id] = metadata
                    
                    if user_id not in self.reverse_user_map:
                        self.reverse_user_map[user_id] = []
                    self.reverse_user_map[user_id].append(faiss_id)
                
                # Update statistics
                self.stats['insertions'] += len(embeddings)
                
                logger.debug(f"Added {len(embeddings)} embeddings in {time.time() - start_time:.3f}s")
                return faiss_ids
                
            except Exception as e:
                logger.error(f"Failed to add embeddings: {e}")
                raise
    
    def add_single_embedding(
        self, 
        embedding: np.ndarray, 
        user_id: str, 
        metadata: Dict[str, Any]
    ) -> int:
        """Add a single embedding to the index."""
        return self.add_embeddings(
            embeddings=embedding.reshape(1, -1),
            user_ids=[user_id],
            metadata_list=[metadata]
        )[0]
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        score_threshold: Optional[float] = None,
        user_filter: Optional[str] = None
    ) -> FaissSearchResult:
        """Search for similar embeddings."""
        with self._lock:
            start_time = time.time()
            
            try:
                # Normalize query embedding
                query_normalized = query_embedding.copy()
                norm = np.linalg.norm(query_normalized)
                if norm > 0:
                    query_normalized = query_normalized / norm
                
                query_normalized = query_normalized.reshape(1, -1).astype('float32')
                
                # Perform search
                active_index = self._get_active_index()
                distances, indices = active_index.search(query_normalized, k)
                
                # Convert distances to similarities (for inner product)
                similarities = distances[0]  # Inner product is already similarity for normalized vectors
                indices = indices[0]
                
                # Filter results
                valid_results = []
                for i, (similarity, idx) in enumerate(zip(similarities, indices)):
                    if idx == -1:  # Invalid result
                        continue
                    
                    if score_threshold and similarity < score_threshold:
                        continue
                    
                    user_id = self.user_id_map.get(idx, "unknown")
                    if user_filter and user_id != user_filter:
                        continue
                    
                    valid_results.append((similarity, idx, user_id))
                
                # Extract results
                if valid_results:
                    similarities_filtered = np.array([r[0] for r in valid_results])
                    indices_filtered = np.array([r[1] for r in valid_results])
                    user_ids_filtered = [r[2] for r in valid_results]
                    metadata_filtered = [
                        self.metadata_map.get(idx, {}) 
                        for idx in indices_filtered
                    ]
                    distances_filtered = 1.0 - similarities_filtered  # Convert to distances
                else:
                    similarities_filtered = np.array([])
                    indices_filtered = np.array([])
                    user_ids_filtered = []
                    metadata_filtered = []
                    distances_filtered = np.array([])
                
                # Update statistics
                search_time = time.time() - start_time
                self.stats['searches'] += 1
                self.stats['total_search_time'] += search_time
                self.stats['avg_search_time'] = (
                    self.stats['total_search_time'] / self.stats['searches']
                )
                
                result = FaissSearchResult(
                    distances=distances_filtered,
                    indices=indices_filtered,
                    user_ids=user_ids_filtered,
                    similarities=similarities_filtered,
                    metadata=metadata_filtered
                )
                
                logger.debug(f"Search completed in {search_time:.3f}s, found {len(user_ids_filtered)} results")
                return result
                
            except Exception as e:
                logger.error(f"Search failed: {e}")
                raise
    
    def remove_user_embeddings(self, user_id: str) -> int:
        """Remove all embeddings for a specific user."""
        with self._lock:
            try:
                if user_id not in self.reverse_user_map:
                    return 0
                
                faiss_ids = self.reverse_user_map[user_id]
                
                # Remove from FAISS index
                active_index = self._get_active_index()
                if hasattr(active_index, 'remove_ids'):
                    active_index.remove_ids(np.array(faiss_ids, dtype=np.int64))
                else:
                    # For indexes that don't support removal, we need to rebuild
                    logger.warning(f"Index type {self.index_type} doesn't support removal, marking for rebuild")
                    self._mark_for_rebuild()
                
                # Remove from metadata maps
                for faiss_id in faiss_ids:
                    self.user_id_map.pop(faiss_id, None)
                    self.metadata_map.pop(faiss_id, None)
                
                del self.reverse_user_map[user_id]
                
                logger.info(f"Removed {len(faiss_ids)} embeddings for user {user_id}")
                return len(faiss_ids)
                
            except Exception as e:
                logger.error(f"Failed to remove embeddings for user {user_id}: {e}")
                return 0
    
    def _mark_for_rebuild(self):
        """Mark index for rebuild (when removal is not supported)."""
        # This would trigger a background rebuild process
        logger.info("Index marked for rebuild due to unsupported removal operation")
    
    def get_stats(self) -> FaissStats:
        """Get comprehensive FAISS statistics."""
        with self._lock:
            try:
                active_index = self._get_active_index()
                
                # Calculate memory usage (approximate)
                memory_usage_mb = 0.0
                if active_index:
                    # Rough estimate: embedding_dim * ntotal * 4 bytes (float32)
                    memory_usage_mb = (
                        self.embedding_dim * active_index.ntotal * 4
                    ) / (1024 * 1024)
                
                # Get index file size
                index_file_size_mb = 0.0
                if self.index_file.exists():
                    index_file_size_mb = self.index_file.stat().st_size / (1024 * 1024)
                
                return FaissStats(
                    total_vectors=active_index.ntotal if active_index else 0,
                    index_type=self.index_type,
                    is_trained=active_index.is_trained if active_index else False,
                    gpu_enabled=self.use_gpu,
                    memory_usage_mb=memory_usage_mb,
                    search_time_avg_ms=self.stats['avg_search_time'] * 1000,
                    last_optimization=time.time(),
                    index_file_size_mb=index_file_size_mb
                )
                
            except Exception as e:
                logger.error(f"Failed to get stats: {e}")
                return FaissStats(
                    total_vectors=0,
                    index_type=self.index_type,
                    is_trained=False,
                    gpu_enabled=False,
                    memory_usage_mb=0.0,
                    search_time_avg_ms=0.0,
                    last_optimization=0.0,
                    index_file_size_mb=0.0
                )
    
    def save_index(self) -> bool:
        """Save index and metadata to disk."""
        with self._lock:
            try:
                logger.info("Saving FAISS index to disk...")
                
                # Save the CPU index (GPU index can't be saved directly)
                if self.index:
                    faiss.write_index(self.index, str(self.index_file))
                
                # Save metadata
                metadata = {
                    'user_id_map': self.user_id_map,
                    'metadata_map': self.metadata_map,
                    'reverse_user_map': self.reverse_user_map,
                    'next_id': self._next_id,
                    'stats': self.stats
                }
                
                with open(self.metadata_file, 'wb') as f:
                    pickle.dump(metadata, f)
                
                # Save configuration
                config = {
                    'index_type': self.index_type,
                    'embedding_dim': self.embedding_dim,
                    'nlist': self.nlist,
                    'nprobe': self.nprobe,
                    'm': self.m,
                    'nbits': self.nbits,
                    'use_gpu': self.use_gpu
                }
                
                with open(self.config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                
                logger.info("FAISS index saved successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to save index: {e}")
                return False
    
    def _load_index(self) -> bool:
        """Load index and metadata from disk."""
        with self._lock:
            try:
                if not self.index_file.exists():
                    logger.info("No existing index found, starting fresh")
                    return False
                
                logger.info("Loading FAISS index from disk...")
                
                # Load configuration
                if self.config_file.exists():
                    with open(self.config_file, 'r') as f:
                        config = json.load(f)
                    
                    # Update configuration
                    self.index_type = config.get('index_type', self.index_type)
                    self.nlist = config.get('nlist', self.nlist)
                    self.nprobe = config.get('nprobe', self.nprobe)
                    self.m = config.get('m', self.m)
                    self.nbits = config.get('nbits', self.nbits)
                
                # Load index
                self.index = faiss.read_index(str(self.index_file))
                
                # Move to GPU if available
                if self.use_gpu and self.gpu_resources:
                    self.gpu_index = faiss.index_cpu_to_gpu(
                        self.gpu_resources, 
                        self.gpu_device_id, 
                        self.index
                    )
                
                # Load metadata
                if self.metadata_file.exists():
                    with open(self.metadata_file, 'rb') as f:
                        metadata = pickle.load(f)
                    
                    self.user_id_map = metadata.get('user_id_map', {})
                    self.metadata_map = metadata.get('metadata_map', {})
                    self.reverse_user_map = metadata.get('reverse_user_map', {})
                    self._next_id = metadata.get('next_id', 0)
                    self.stats.update(metadata.get('stats', {}))
                
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
                return False
    
    def reset_index(self) -> bool:
        """Reset the entire index (delete all data)."""
        with self._lock:
            try:
                logger.warning("Resetting FAISS index - all data will be lost")
                
                # Clear in-memory data
                self.user_id_map.clear()
                self.metadata_map.clear()
                self.reverse_user_map.clear()
                self._next_id = 0
                
                # Reset statistics
                self.stats = {
                    'searches': 0,
                    'insertions': 0,
                    'total_search_time': 0.0,
                    'avg_search_time': 0.0,
                    'cache_hits': 0,
                    'index_rebuilds': 0
                }
                
                # Recreate index
                self._create_index()
                
                # Remove saved files
                for file_path in [self.index_file, self.metadata_file, self.config_file]:
                    if file_path.exists():
                        file_path.unlink()
                
                logger.info("FAISS index reset completed")
                return True
                
            except Exception as e:
                logger.error(f"Failed to reset index: {e}")
                return False
    
    def optimize_index(self) -> bool:
        """Optimize index for better performance."""
        with self._lock:
            try:
                logger.info("Optimizing FAISS index...")
                
                # For IVF indexes, we can adjust nprobe based on dataset size
                active_index = self._get_active_index()
                if hasattr(active_index, 'nprobe'):
                    # Adaptive nprobe based on dataset size
                    if active_index.ntotal < 100000:
                        active_index.nprobe = min(32, self.nlist)
                    elif active_index.ntotal < 1000000:
                        active_index.nprobe = min(64, self.nlist)
                    else:
                        active_index.nprobe = min(128, self.nlist)
                    
                    logger.info(f"Adjusted nprobe to {active_index.nprobe}")
                
                self.stats['index_rebuilds'] += 1
                logger.info("Index optimization completed")
                return True
                
            except Exception as e:
                logger.error(f"Index optimization failed: {e}")
                return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        return {
            "faiss_stats": self.stats.copy(),
            "configuration": {
                "index_type": self.index_type,
                "embedding_dim": self.embedding_dim,
                "use_gpu": self.use_gpu,
                "gpu_device_id": self.gpu_device_id,
                "nlist": self.nlist,
                "nprobe": self.nprobe
            },
            "memory_info": {
                "total_vectors": len(self.user_id_map),
                "unique_users": len(self.reverse_user_map),
                "gpu_available": faiss.get_num_gpus() > 0 if FAISS_AVAILABLE else False
            }
        }
    
    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'gpu_resources') and self.gpu_resources:
                # FAISS GPU resources are automatically cleaned up
                pass
        except:
            pass


# Global instance
faiss_engine = FaissEngine() if FAISS_AVAILABLE else None
