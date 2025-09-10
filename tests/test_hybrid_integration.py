"""
Test script for hybrid vector store integration with production scenarios.
Tests Qdrant + FAISS hybrid architecture for 10M+ user scale.
"""

import asyncio
import time
import numpy as np
import logging
from typing import List, Dict, Any
import json
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.hybrid_vector_store import hybrid_vector_store
from src.config.settings import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridVectorStoreTestSuite:
    """Comprehensive test suite for hybrid vector store."""
    
    def __init__(self):
        self.test_embeddings: List[np.ndarray] = []
        self.test_users: List[str] = []
        self.test_metadata: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {}
        
    async def setup_test_data(self, num_users: int = 1000):
        """Generate test embeddings and metadata."""
        logger.info(f"Generating {num_users} test embeddings...")
        
        np.random.seed(42)  # For reproducible results
        
        for i in range(num_users):
            # Generate normalized 512-dimensional embeddings
            embedding = np.random.randn(512).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            
            user_id = f"test_user_{i:06d}"
            metadata = {
                "name": f"Test User {i}",
                "timestamp": time.time(),
                "test_batch": "production_test",
                "user_index": i
            }
            
            self.test_embeddings.append(embedding)
            self.test_users.append(user_id)
            self.test_metadata.append(metadata)
        
        logger.info(f"Generated {len(self.test_embeddings)} test embeddings")
    
    async def test_initialization(self) -> bool:
        """Test hybrid vector store initialization."""
        logger.info("Testing hybrid vector store initialization...")
        
        try:
            success = await hybrid_vector_store.initialize()
            if success:
                stats = await hybrid_vector_store.get_stats()
                logger.info(f"Initialization successful - Qdrant: {stats.qdrant_health}, FAISS: {stats.faiss_health}")
                return True
            else:
                logger.error("Initialization failed")
                return False
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False
    
    async def test_bulk_enrollment(self, batch_size: int = 100) -> Dict[str, Any]:
        """Test bulk user enrollment performance."""
        logger.info(f"Testing bulk enrollment with {len(self.test_users)} users (batch size: {batch_size})...")
        
        start_time = time.time()
        successful_enrollments = 0
        failed_enrollments = 0
        enrollment_times = []
        
        for i in range(0, len(self.test_users), batch_size):
            batch_users = self.test_users[i:i+batch_size]
            batch_embeddings = self.test_embeddings[i:i+batch_size]
            batch_metadata = self.test_metadata[i:i+batch_size]
            
            batch_start = time.time()
            
            # Process batch
            for user_id, embedding, metadata in zip(batch_users, batch_embeddings, batch_metadata):
                try:
                    await hybrid_vector_store.add_embedding(user_id, embedding, metadata)
                    successful_enrollments += 1
                except Exception as e:
                    logger.error(f"Failed to enroll {user_id}: {e}")
                    failed_enrollments += 1
            
            batch_time = time.time() - batch_start
            enrollment_times.append(batch_time)
            
            if i % (batch_size * 10) == 0:  # Log progress every 10 batches
                logger.info(f"Processed {i + len(batch_users)} users...")
        
        total_time = time.time() - start_time
        
        results = {
            "total_users": len(self.test_users),
            "successful_enrollments": successful_enrollments,
            "failed_enrollments": failed_enrollments,
            "total_time_seconds": total_time,
            "avg_time_per_user_ms": (total_time / successful_enrollments) * 1000 if successful_enrollments > 0 else 0,
            "throughput_users_per_second": successful_enrollments / total_time if total_time > 0 else 0,
            "avg_batch_time_seconds": np.mean(enrollment_times) if enrollment_times else 0
        }
        
        logger.info(f"Bulk enrollment completed: {successful_enrollments}/{len(self.test_users)} users in {total_time:.2f}s")
        logger.info(f"Throughput: {results['throughput_users_per_second']:.2f} users/sec")
        
        return results
    
    async def test_search_performance(self, num_queries: int = 100, k: int = 5) -> Dict[str, Any]:
        """Test search performance with different strategies."""
        logger.info(f"Testing search performance with {num_queries} queries...")
        
        # Generate query embeddings (similar to enrolled users but with noise)
        np.random.seed(123)
        query_embeddings = []
        for i in range(num_queries):
            # Take a random enrolled embedding and add noise
            base_idx = np.random.randint(0, len(self.test_embeddings))
            base_embedding = self.test_embeddings[base_idx].copy()
            noise = np.random.randn(512) * 0.1  # Small noise
            query_embedding = base_embedding + noise
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embeddings.append(query_embedding)
        
        strategies = ["hybrid", "faiss_first", "qdrant_only"]
        results = {}
        
        for strategy in strategies:
            logger.info(f"Testing {strategy} search strategy...")
            
            search_times = []
            successful_searches = 0
            total_results = 0
            
            strategy_start = time.time()
            
            for i, query_embedding in enumerate(query_embeddings):
                try:
                    search_start = time.time()
                    search_results = await hybrid_vector_store.search_similar(
                        query_embedding, k=k, strategy=strategy
                    )
                    search_time = time.time() - search_start
                    
                    search_times.append(search_time)
                    successful_searches += 1
                    total_results += len(search_results)
                    
                    if i % 20 == 0:  # Log progress
                        logger.info(f"  Completed {i+1}/{num_queries} searches...")
                        
                except Exception as e:
                    logger.error(f"Search failed for query {i}: {e}")
            
            strategy_time = time.time() - strategy_start
            
            results[strategy] = {
                "successful_searches": successful_searches,
                "total_queries": num_queries,
                "total_time_seconds": strategy_time,
                "avg_search_time_ms": (np.mean(search_times) * 1000) if search_times else 0,
                "p95_search_time_ms": (np.percentile(search_times, 95) * 1000) if search_times else 0,
                "p99_search_time_ms": (np.percentile(search_times, 99) * 1000) if search_times else 0,
                "min_search_time_ms": (np.min(search_times) * 1000) if search_times else 0,
                "max_search_time_ms": (np.max(search_times) * 1000) if search_times else 0,
                "throughput_searches_per_second": successful_searches / strategy_time if strategy_time > 0 else 0,
                "avg_results_per_query": total_results / successful_searches if successful_searches > 0 else 0
            }
            
            logger.info(f"  {strategy}: {results[strategy]['avg_search_time_ms']:.2f}ms avg, "
                       f"{results[strategy]['throughput_searches_per_second']:.2f} searches/sec")
        
        return results
    
    async def test_concurrent_operations(self, num_concurrent: int = 10) -> Dict[str, Any]:
        """Test concurrent search and enrollment operations."""
        logger.info(f"Testing {num_concurrent} concurrent operations...")
        
        async def concurrent_search():
            """Perform concurrent searches."""
            query_embedding = np.random.randn(512).astype(np.float32)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            start_time = time.time()
            results = await hybrid_vector_store.search_similar(query_embedding, k=5)
            search_time = time.time() - start_time
            
            return {"search_time": search_time, "results_count": len(results)}
        
        async def concurrent_enrollment():
            """Perform concurrent enrollment."""
            user_id = f"concurrent_user_{int(time.time() * 1000000)}"
            embedding = np.random.randn(512).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            metadata = {"name": "Concurrent Test User", "timestamp": time.time()}
            
            start_time = time.time()
            await hybrid_vector_store.add_embedding(user_id, embedding, metadata)
            enrollment_time = time.time() - start_time
            
            return {"enrollment_time": enrollment_time, "user_id": user_id}
        
        # Run concurrent operations
        start_time = time.time()
        
        # Mix of search and enrollment operations
        tasks = []
        for i in range(num_concurrent):
            if i % 3 == 0:  # 1/3 enrollments, 2/3 searches
                tasks.append(concurrent_enrollment())
            else:
                tasks.append(concurrent_search())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_ops = 0
        failed_ops = 0
        search_times = []
        enrollment_times = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_ops += 1
                logger.error(f"Concurrent operation failed: {result}")
            else:
                successful_ops += 1
                if "search_time" in result:
                    search_times.append(result["search_time"])
                elif "enrollment_time" in result:
                    enrollment_times.append(result["enrollment_time"])
        
        return {
            "total_operations": num_concurrent,
            "successful_operations": successful_ops,
            "failed_operations": failed_ops,
            "total_time_seconds": total_time,
            "avg_search_time_ms": (np.mean(search_times) * 1000) if search_times else 0,
            "avg_enrollment_time_ms": (np.mean(enrollment_times) * 1000) if enrollment_times else 0,
            "concurrent_throughput_ops_per_second": successful_ops / total_time if total_time > 0 else 0
        }
    
    async def test_system_stats(self) -> Dict[str, Any]:
        """Test system statistics and health checks."""
        logger.info("Testing system statistics...")
        
        try:
            stats = await hybrid_vector_store.get_stats()
            performance_metrics = hybrid_vector_store.get_performance_metrics()
            
            return {
                "hybrid_stats": {
                    "total_vectors": stats.total_vectors,
                    "qdrant_vectors": stats.qdrant_vectors,
                    "faiss_vectors": stats.faiss_vectors,
                    "cache_hit_rate": stats.cache_hit_rate,
                    "avg_search_time_ms": stats.avg_search_time_ms,
                    "qdrant_health": stats.qdrant_health,
                    "faiss_health": stats.faiss_health,
                    "sync_status": stats.sync_status,
                    "memory_usage_mb": stats.memory_usage_mb
                },
                "performance_metrics": performance_metrics
            }
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {"error": str(e)}
    
    async def run_full_test_suite(self, num_users: int = 1000, num_queries: int = 100):
        """Run the complete test suite."""
        logger.info("Starting hybrid vector store test suite...")
        
        # Initialize results
        self.results = {
            "test_config": {
                "num_users": num_users,
                "num_queries": num_queries,
                "embedding_dimension": 512,
                "timestamp": time.time()
            },
            "tests": {}
        }
        
        try:
            # Test 1: Initialization
            init_success = await self.test_initialization()
            self.results["tests"]["initialization"] = {"success": init_success}
            
            if not init_success:
                logger.error("Initialization failed, stopping tests")
                return self.results
            
            # Test 2: Setup test data
            await self.setup_test_data(num_users)
            
            # Test 3: Bulk enrollment
            enrollment_results = await self.test_bulk_enrollment()
            self.results["tests"]["bulk_enrollment"] = enrollment_results
            
            # Test 4: Search performance
            search_results = await self.test_search_performance(num_queries)
            self.results["tests"]["search_performance"] = search_results
            
            # Test 5: Concurrent operations
            concurrent_results = await self.test_concurrent_operations()
            self.results["tests"]["concurrent_operations"] = concurrent_results
            
            # Test 6: System statistics
            stats_results = await self.test_system_stats()
            self.results["tests"]["system_stats"] = stats_results
            
            logger.info("Test suite completed successfully!")
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            self.results["error"] = str(e)
        
        return self.results
    
    def save_results(self, filename: str = "hybrid_test_results.json"):
        """Save test results to JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"Test results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def print_summary(self):
        """Print a summary of test results."""
        if not self.results:
            logger.warning("No test results available")
            return
        
        print("\n" + "="*80)
        print("HYBRID VECTOR STORE TEST RESULTS SUMMARY")
        print("="*80)
        
        config = self.results.get("test_config", {})
        print(f"Test Configuration:")
        print(f"  Users: {config.get('num_users', 'N/A')}")
        print(f"  Queries: {config.get('num_queries', 'N/A')}")
        print(f"  Embedding Dimension: {config.get('embedding_dimension', 'N/A')}")
        
        tests = self.results.get("tests", {})
        
        # Enrollment results
        if "bulk_enrollment" in tests:
            enrollment = tests["bulk_enrollment"]
            print(f"\nBulk Enrollment:")
            print(f"  Success Rate: {enrollment.get('successful_enrollments', 0)}/{enrollment.get('total_users', 0)}")
            print(f"  Throughput: {enrollment.get('throughput_users_per_second', 0):.2f} users/sec")
            print(f"  Avg Time per User: {enrollment.get('avg_time_per_user_ms', 0):.2f}ms")
        
        # Search results
        if "search_performance" in tests:
            search = tests["search_performance"]
            print(f"\nSearch Performance:")
            for strategy, results in search.items():
                print(f"  {strategy.upper()}:")
                print(f"    Avg Time: {results.get('avg_search_time_ms', 0):.2f}ms")
                print(f"    P95 Time: {results.get('p95_search_time_ms', 0):.2f}ms")
                print(f"    Throughput: {results.get('throughput_searches_per_second', 0):.2f} searches/sec")
        
        # System stats
        if "system_stats" in tests and "hybrid_stats" in tests["system_stats"]:
            stats = tests["system_stats"]["hybrid_stats"]
            print(f"\nSystem Statistics:")
            print(f"  Total Vectors: {stats.get('total_vectors', 0)}")
            print(f"  Qdrant Vectors: {stats.get('qdrant_vectors', 0)}")
            print(f"  FAISS Vectors: {stats.get('faiss_vectors', 0)}")
            print(f"  Cache Hit Rate: {stats.get('cache_hit_rate', 0):.2%}")
            print(f"  Qdrant Health: {stats.get('qdrant_health', False)}")
            print(f"  FAISS Health: {stats.get('faiss_health', False)}")
        
        print("="*80)


async def main():
    """Main test execution."""
    # Configuration
    NUM_USERS = 1000  # Start with 1K users for testing
    NUM_QUERIES = 100
    
    # Create test suite
    test_suite = HybridVectorStoreTestSuite()
    
    # Run tests
    results = await test_suite.run_full_test_suite(NUM_USERS, NUM_QUERIES)
    
    # Save and display results
    test_suite.save_results("hybrid_integration_test_results.json")
    test_suite.print_summary()
    
    # Clean up
    try:
        await hybrid_vector_store.close()
    except Exception as e:
        logger.error(f"Cleanup error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
