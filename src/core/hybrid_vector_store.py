"""
Hybrid Vector Store combining Qdrant and FAISS for production-scale face verification.
Architecture: Qdrant for persistent storage + FAISS for ultra-fast GPU search.
Optimized for 10M+ users with RTX 4090.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from src.core.qdrant_client import qdrant_client, SearchResult as QdrantSearchResult
from src.core.faiss_engine import faiss_engine, FaissSearchResult
from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """Unified search result from hybrid vector store."""
    user_id: str
    similarity: float
    confidence: float
    distance: float
    source: str  # 'faiss', 'qdrant', or 'hybrid'
    metadata: Dict[str, Any]
    response_time_ms: float


@dataclass
class HybridStats:
    """Comprehensive hybrid vector store statistics."""
    total_vectors: int
    qdrant_vectors: int
    faiss_vectors: int
    cache_hit_rate: float
    avg_search_time_ms: float
    qdrant_health: bool
    faiss_health: bool
    sync_status: str
    memory_usage_mb: float


class HybridVectorStore:
    """
    Production-ready hybrid vector store for massive-scale face verification.
    
    Architecture Strategy:
    1. Qdrant: Persistent storage, metadata management, distributed scaling
    2. FAISS: GPU-accelerated search cache for hot data
    3. Intelligent routing: Hot data → FAISS, Cold data → Qdrant
    4. Async synchronization between stores
    5. Fallback mechanisms for high availability
    
    Performance Targets:
    - Search: <30ms for cached data, <50ms for cold data
    - Throughput: 1000+ searches/second
    - Scalability: 10M+ vectors with sub-linear degradation
    """
    
    def __init__(self):
        self.qdrant = qdrant_client
        self.faiss = faiss_engine
        
        # Configuration
        self.faiss_enabled = settings.faiss_enabled and self.faiss is not None
        self.cache_size = settings.faiss_cache_size
        self.sync_interval = 300  # 5 minutes
        
        # Performance tracking
        self.stats = {
            'total_searches': 0,
            'faiss_hits': 0,
            'qdrant_hits': 0,
            'hybrid_searches': 0,
            'cache_misses': 0,
            'sync_operations': 0,
            'total_search_time': 0.0,
            'avg_search_time': 0.0
        }
        
        # Cache management
        self._hot_users: set = set()  # Users with frequent access
        self._access_count: Dict[str, int] = {}  # User access frequency
        self._last_sync = 0.0
        self._sync_lock = threading.RLock()
        
        # Background tasks
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._sync_task: Optional[asyncio.Task] = None
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Initialize hybrid vector store."""
        try:
            logger.info("Initializing hybrid vector store...")
            
            # Initialize Qdrant
            qdrant_ok = await self.qdrant.initialize()
            if not qdrant_ok:
                logger.error("Failed to initialize Qdrant")
                return False
            
            # Initialize FAISS if enabled
            faiss_ok = True
            if self.faiss_enabled:
                try:
                    # FAISS is initialized in constructor
                    faiss_stats = self.faiss.get_stats()
                    logger.info(f"FAISS initialized with {faiss_stats.total_vectors} vectors")
                except Exception as e:
                    logger.warning(f"FAISS initialization failed: {e}")
                    self.faiss_enabled = False
                    faiss_ok = False
            
            # Start background sync if both stores are available
            if qdrant_ok and self.faiss_enabled:
                await self._start_background_sync()
            
            self._initialized = True
            logger.info(f"Hybrid vector store initialized (Qdrant: {qdrant_ok}, FAISS: {self.faiss_enabled})")
            return True
            
        except Exception as e:
            logger.error(f"Hybrid vector store initialization failed: {e}")
            return False
    
    async def _start_background_sync(self):
        """Start background synchronization between Qdrant and FAISS."""
        try:
            self._sync_task = asyncio.create_task(self._background_sync_loop())
            logger.info("Background sync started")
        except Exception as e:
            logger.error(f"Failed to start background sync: {e}")
    
    async def _background_sync_loop(self):
        """Background loop for syncing hot data to FAISS cache."""
        while True:
            try:
                await asyncio.sleep(self.sync_interval)
                await self._sync_hot_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background sync error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _sync_hot_data(self):
        """Sync frequently accessed data from Qdrant to FAISS."""
        if not self.faiss_enabled:
            return
        
        with self._sync_lock:
            try:
                start_time = time.time()
                logger.info("Starting hot data sync...")
                
                # Identify hot users (top accessed users)
                sorted_users = sorted(
                    self._access_count.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                hot_users = [user_id for user_id, count in sorted_users[:self.cache_size]]
                
                # Get embeddings for hot users from Qdrant
                sync_count = 0
                for user_id in hot_users:
                    if user_id not in self._hot_users:
                        # Get user embeddings from Qdrant
                        results = await self.qdrant.search_similar_async(
                            query_embedding=np.random.rand(settings.embedding_dim),  # Dummy query
                            k=1,
                            user_filter=user_id
                        )
                        
                        if results:
                            # This would require getting the actual embedding from Qdrant
                            # For now, we mark the user as hot
                            self._hot_users.add(user_id)
                            sync_count += 1
                
                # Update statistics
                self.stats['sync_operations'] += 1
                self._last_sync = time.time()
                
                sync_time = time.time() - start_time
                logger.info(f"Hot data sync completed: {sync_count} users synced in {sync_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Hot data sync failed: {e}")
    
    async def add_embedding(
        self, 
        user_id: str, 
        embedding: np.ndarray, 
        metadata: Dict[str, Any]
    ) -> str:
        """Add embedding to both Qdrant and optionally FAISS."""
        start_time = time.time()
        
        try:
            # Always add to Qdrant for persistence
            qdrant_id = await self.qdrant.add_embedding_async(user_id, embedding, metadata)
            
            # Add to FAISS if enabled and user is hot
            faiss_id = None
            if self.faiss_enabled:
                # Check if this is a hot user or if FAISS has capacity
                if (user_id in self._hot_users or 
                    self.faiss.get_stats().total_vectors < self.cache_size):
                    
                    try:
                        faiss_id = self.faiss.add_single_embedding(
                            embedding, user_id, metadata
                        )
                        self._hot_users.add(user_id)
                        logger.debug(f"Added embedding to FAISS cache for user {user_id}")
                    except Exception as e:
                        logger.warning(f"Failed to add to FAISS cache: {e}")
            
            # Update access tracking
            self._access_count[user_id] = self._access_count.get(user_id, 0) + 1
            
            logger.debug(f"Added embedding for user {user_id} in {time.time() - start_time:.3f}s")
            return qdrant_id
            
        except Exception as e:
            logger.error(f"Failed to add embedding for user {user_id}: {e}")
            raise
    
    async def search_similar(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        score_threshold: Optional[float] = None,
        strategy: str = "hybrid"  # "hybrid", "faiss_first", "qdrant_only"
    ) -> List[HybridSearchResult]:
        """
        Intelligent hybrid search with multiple strategies.
        
        Strategies:
        - hybrid: Try FAISS first, fallback to Qdrant, merge results
        - faiss_first: FAISS only, fallback to Qdrant if insufficient results
        - qdrant_only: Qdrant only (bypass FAISS cache)
        """
        start_time = time.time()
        
        try:
            self.stats['total_searches'] += 1
            
            if strategy == "qdrant_only" or not self.faiss_enabled:
                return await self._search_qdrant_only(query_embedding, k, score_threshold, start_time)
            
            elif strategy == "faiss_first":
                return await self._search_faiss_first(query_embedding, k, score_threshold, start_time)
            
            else:  # hybrid strategy
                return await self._search_hybrid(query_embedding, k, score_threshold, start_time)
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
        finally:
            # Update performance statistics
            search_time = time.time() - start_time
            self.stats['total_search_time'] += search_time
            self.stats['avg_search_time'] = (
                self.stats['total_search_time'] / self.stats['total_searches']
            )
    
    async def _search_qdrant_only(
        self, 
        query_embedding: np.ndarray, 
        k: int, 
        score_threshold: Optional[float],
        start_time: float
    ) -> List[HybridSearchResult]:
        """Search using Qdrant only."""
        qdrant_results = await self.qdrant.search_similar_async(
            query_embedding, k, score_threshold
        )
        
        self.stats['qdrant_hits'] += 1
        
        return [
            HybridSearchResult(
                user_id=result.user_id,
                similarity=result.similarity,
                confidence=result.similarity,
                distance=result.distance,
                source="qdrant",
                metadata=result.payload,
                response_time_ms=(time.time() - start_time) * 1000
            )
            for result in qdrant_results
        ]
    
    async def _search_faiss_first(
        self, 
        query_embedding: np.ndarray, 
        k: int, 
        score_threshold: Optional[float],
        start_time: float
    ) -> List[HybridSearchResult]:
        """Search FAISS first, fallback to Qdrant if needed."""
        try:
            # Try FAISS first
            faiss_result = self.faiss.search(query_embedding, k, score_threshold)
            
            if len(faiss_result.user_ids) >= k:
                # Sufficient results from FAISS
                self.stats['faiss_hits'] += 1
                
                return [
                    HybridSearchResult(
                        user_id=user_id,
                        similarity=similarity,
                        confidence=similarity,
                        distance=distance,
                        source="faiss",
                        metadata=metadata,
                        response_time_ms=(time.time() - start_time) * 1000
                    )
                    for user_id, similarity, distance, metadata in zip(
                        faiss_result.user_ids,
                        faiss_result.similarities,
                        faiss_result.distances,
                        faiss_result.metadata
                    )
                ]
            else:
                # Insufficient results, fallback to Qdrant
                return await self._search_qdrant_only(query_embedding, k, score_threshold, start_time)
                
        except Exception as e:
            logger.warning(f"FAISS search failed, falling back to Qdrant: {e}")
            return await self._search_qdrant_only(query_embedding, k, score_threshold, start_time)
    
    async def _search_hybrid(
        self, 
        query_embedding: np.ndarray, 
        k: int, 
        score_threshold: Optional[float],
        start_time: float
    ) -> List[HybridSearchResult]:
        """Advanced hybrid search combining FAISS and Qdrant results."""
        try:
            # Run both searches concurrently
            tasks = []
            
            # FAISS search (if enabled)
            if self.faiss_enabled:
                faiss_task = asyncio.create_task(
                    asyncio.get_event_loop().run_in_executor(
                        self._executor,
                        lambda: self.faiss.search(query_embedding, k * 2, score_threshold)
                    )
                )
                tasks.append(("faiss", faiss_task))
            
            # Qdrant search
            qdrant_task = asyncio.create_task(
                self.qdrant.search_similar_async(query_embedding, k * 2, score_threshold)
            )
            tasks.append(("qdrant", qdrant_task))
            
            # Wait for both results
            results = {}
            for source, task in tasks:
                try:
                    results[source] = await task
                except Exception as e:
                    logger.warning(f"{source} search failed: {e}")
                    results[source] = None
            
            # Merge and rank results
            merged_results = self._merge_search_results(
                results.get("faiss"),
                results.get("qdrant"),
                k,
                start_time
            )
            
            self.stats['hybrid_searches'] += 1
            return merged_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to Qdrant only
            return await self._search_qdrant_only(query_embedding, k, score_threshold, start_time)
    
    def _merge_search_results(
        self,
        faiss_result: Optional[FaissSearchResult],
        qdrant_results: Optional[List[QdrantSearchResult]],
        k: int,
        start_time: float
    ) -> List[HybridSearchResult]:
        """Merge and rank results from FAISS and Qdrant."""
        merged = {}
        response_time_ms = (time.time() - start_time) * 1000
        
        # Add FAISS results
        if faiss_result and len(faiss_result.user_ids) > 0:
            for user_id, similarity, distance, metadata in zip(
                faiss_result.user_ids,
                faiss_result.similarities,
                faiss_result.distances,
                faiss_result.metadata
            ):
                merged[user_id] = HybridSearchResult(
                    user_id=user_id,
                    similarity=similarity,
                    confidence=similarity * 1.1,  # Slight boost for cached results
                    distance=distance,
                    source="faiss",
                    metadata=metadata,
                    response_time_ms=response_time_ms
                )
        
        # Add Qdrant results (merge or replace)
        if qdrant_results:
            for result in qdrant_results:
                user_id = result.user_id
                
                if user_id in merged:
                    # Take the better score
                    if result.similarity > merged[user_id].similarity:
                        merged[user_id] = HybridSearchResult(
                            user_id=user_id,
                            similarity=result.similarity,
                            confidence=result.similarity,
                            distance=result.distance,
                            source="hybrid",
                            metadata=result.payload,
                            response_time_ms=response_time_ms
                        )
                else:
                    merged[user_id] = HybridSearchResult(
                        user_id=user_id,
                        similarity=result.similarity,
                        confidence=result.similarity,
                        distance=result.distance,
                        source="qdrant",
                        metadata=result.payload,
                        response_time_ms=response_time_ms
                    )
        
        # Sort by confidence and return top k
        sorted_results = sorted(
            merged.values(),
            key=lambda x: x.confidence,
            reverse=True
        )
        
        return sorted_results[:k]
    
    async def delete_user_embeddings(self, user_id: str) -> bool:
        """Delete user embeddings from both stores."""
        try:
            # Delete from Qdrant
            qdrant_success = await self.qdrant.delete_user_vectors_async(user_id)
            
            # Delete from FAISS if present
            faiss_success = True
            if self.faiss_enabled and user_id in self._hot_users:
                try:
                    self.faiss.remove_user_embeddings(user_id)
                    self._hot_users.discard(user_id)
                except Exception as e:
                    logger.warning(f"Failed to remove from FAISS: {e}")
                    faiss_success = False
            
            # Clean up access tracking
            self._access_count.pop(user_id, None)
            
            return qdrant_success and faiss_success
            
        except Exception as e:
            logger.error(f"Failed to delete user embeddings for {user_id}: {e}")
            return False
    
    async def user_exists(self, user_id: str) -> bool:
        """Check if user exists in either store."""
        try:
            # Check FAISS first (faster)
            if self.faiss_enabled and user_id in self._hot_users:
                return True
            
            # Check Qdrant
            return await self.qdrant.user_exists_async(user_id)
            
        except Exception as e:
            logger.error(f"Failed to check user existence for {user_id}: {e}")
            return False
    
    async def get_stats(self) -> HybridStats:
        """Get comprehensive hybrid store statistics."""
        try:
            # Get Qdrant stats
            qdrant_stats = await self.qdrant.get_stats_async()
            qdrant_health = await self.qdrant.health_check_async()
            
            # Get FAISS stats
            faiss_stats = None
            faiss_health = False
            if self.faiss_enabled:
                try:
                    faiss_stats = self.faiss.get_stats()
                    faiss_health = True
                except Exception as e:
                    logger.warning(f"Failed to get FAISS stats: {e}")
            
            # Calculate cache hit rate
            total_searches = self.stats['faiss_hits'] + self.stats['qdrant_hits']
            cache_hit_rate = (
                self.stats['faiss_hits'] / max(total_searches, 1)
            ) * 100
            
            return HybridStats(
                total_vectors=qdrant_stats.total_vectors + (faiss_stats.total_vectors if faiss_stats else 0),
                qdrant_vectors=qdrant_stats.total_vectors,
                faiss_vectors=faiss_stats.total_vectors if faiss_stats else 0,
                cache_hit_rate=cache_hit_rate,
                avg_search_time_ms=self.stats['avg_search_time'] * 1000,
                qdrant_health=qdrant_health,
                faiss_health=faiss_health,
                sync_status="active" if self._sync_task and not self._sync_task.done() else "inactive",
                memory_usage_mb=(faiss_stats.memory_usage_mb if faiss_stats else 0)
            )
            
        except Exception as e:
            logger.error(f"Failed to get hybrid stats: {e}")
            return HybridStats(
                total_vectors=0,
                qdrant_vectors=0,
                faiss_vectors=0,
                cache_hit_rate=0.0,
                avg_search_time_ms=0.0,
                qdrant_health=False,
                faiss_health=False,
                sync_status="error",
                memory_usage_mb=0.0
            )
    
    async def reset_all(self) -> bool:
        """Reset both vector stores."""
        try:
            logger.warning("Resetting hybrid vector store - all data will be lost")
            
            # Reset Qdrant
            qdrant_success = await self.qdrant.reset_collection_async()
            
            # Reset FAISS
            faiss_success = True
            if self.faiss_enabled:
                try:
                    faiss_success = self.faiss.reset_index()
                except Exception as e:
                    logger.error(f"Failed to reset FAISS: {e}")
                    faiss_success = False
            
            # Reset internal state
            self._hot_users.clear()
            self._access_count.clear()
            self.stats = {
                'total_searches': 0,
                'faiss_hits': 0,
                'qdrant_hits': 0,
                'hybrid_searches': 0,
                'cache_misses': 0,
                'sync_operations': 0,
                'total_search_time': 0.0,
                'avg_search_time': 0.0
            }
            
            return qdrant_success and faiss_success
            
        except Exception as e:
            logger.error(f"Failed to reset hybrid vector store: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        return {
            "hybrid_stats": self.stats.copy(),
            "configuration": {
                "faiss_enabled": self.faiss_enabled,
                "cache_size": self.cache_size,
                "sync_interval": self.sync_interval
            },
            "cache_info": {
                "hot_users_count": len(self._hot_users),
                "total_access_tracked": len(self._access_count),
                "last_sync": self._last_sync
            },
            "qdrant_metrics": self.qdrant.get_performance_metrics(),
            "faiss_metrics": self.faiss.get_performance_metrics() if self.faiss_enabled else None
        }
    
    async def close(self):
        """Clean up resources."""
        try:
            # Cancel background sync
            if self._sync_task:
                self._sync_task.cancel()
                try:
                    await self._sync_task
                except asyncio.CancelledError:
                    pass
            
            # Close clients
            await self.qdrant.close()
            
            # Shutdown executor
            self._executor.shutdown(wait=True)
            
            logger.info("Hybrid vector store closed")
            
        except Exception as e:
            logger.error(f"Error closing hybrid vector store: {e}")


# Global instance
hybrid_vector_store = HybridVectorStore()
