"""Performance benchmark script for comparing vector database implementations."""

import time
import requests
import numpy as np
import json
import logging
from typing import Dict, List, Any
import concurrent.futures
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDBBenchmark:
    """Benchmark suite for vector database performance comparison."""
    
    def __init__(self):
        self.services = {
            'FAISS-GPU': 'http://faiss-gpu-optimized:8001',
            'ChromaDB': 'http://chromadb-service:8002',
            'Qdrant': 'http://qdrant-service:8003'
        }
        self.embedding_dim = 512
        self.test_results = {}
    
    def generate_test_data(self, num_vectors: int) -> List[Dict]:
        """Generate test embedding data."""
        test_data = []
        for i in range(num_vectors):
            embedding = np.random.rand(self.embedding_dim).astype(np.float32)
            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            
            test_data.append({
                'user_id': f'test_user_{i}',
                'embedding': embedding.tolist(),
                'metadata': {
                    'test_id': i,
                    'category': f'category_{i % 10}',
                    'timestamp': time.time()
                }
            })
        
        return test_data
    
    def wait_for_service(self, service_url: str, timeout: int = 60) -> bool:
        """Wait for service to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{service_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info(f"Service {service_url} is ready")
                    return True
            except:
                pass
            time.sleep(2)
        
        logger.warning(f"Service {service_url} not ready after {timeout}s")
        return False
    
    def benchmark_insert_performance(self, service_url: str, test_data: List[Dict]) -> Dict[str, Any]:
        """Benchmark vector insertion performance."""
        logger.info(f"Benchmarking insert performance for {service_url}")
        
        # Single insert benchmark
        single_times = []
        for i in range(min(100, len(test_data))):
            start_time = time.time()
            try:
                response = requests.post(
                    f"{service_url}/vectors/add",
                    json={
                        'user_id': test_data[i]['user_id'],
                        'embedding': test_data[i]['embedding'],
                        'metadata': test_data[i]['metadata']
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    single_times.append((time.time() - start_time) * 1000)
            except Exception as e:
                logger.error(f"Single insert failed: {e}")
        
        # Batch insert benchmark (if supported)
        batch_times = []
        batch_sizes = [10, 50, 100]
        
        for batch_size in batch_sizes:
            if len(test_data) >= batch_size:
                batch_data = test_data[:batch_size]
                start_time = time.time()
                try:
                    response = requests.post(
                        f"{service_url}/vectors/batch-add",
                        json={'vectors': batch_data},
                        timeout=60
                    )
                    if response.status_code == 200:
                        batch_time = (time.time() - start_time) * 1000
                        batch_times.append({
                            'batch_size': batch_size,
                            'total_time_ms': batch_time,
                            'time_per_vector_ms': batch_time / batch_size
                        })
                except Exception as e:
                    logger.error(f"Batch insert failed for size {batch_size}: {e}")
        
        return {
            'single_insert': {
                'avg_time_ms': statistics.mean(single_times) if single_times else 0,
                'min_time_ms': min(single_times) if single_times else 0,
                'max_time_ms': max(single_times) if single_times else 0,
                'samples': len(single_times)
            },
            'batch_insert': batch_times
        }
    
    def benchmark_search_performance(self, service_url: str, num_searches: int = 100) -> Dict[str, Any]:
        """Benchmark vector search performance."""
        logger.info(f"Benchmarking search performance for {service_url}")
        
        search_times = []
        successful_searches = 0
        
        for i in range(num_searches):
            # Generate random query
            query_embedding = np.random.rand(self.embedding_dim).astype(np.float32)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            start_time = time.time()
            try:
                response = requests.post(
                    f"{service_url}/vectors/search",
                    json={
                        'embedding': query_embedding.tolist(),
                        'k': 5,
                        'threshold': 0.5
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    search_time = (time.time() - start_time) * 1000
                    search_times.append(search_time)
                    successful_searches += 1
                    
            except Exception as e:
                logger.error(f"Search failed: {e}")
        
        if search_times:
            return {
                'avg_search_time_ms': statistics.mean(search_times),
                'min_search_time_ms': min(search_times),
                'max_search_time_ms': max(search_times),
                'p95_search_time_ms': np.percentile(search_times, 95),
                'p99_search_time_ms': np.percentile(search_times, 99),
                'successful_searches': successful_searches,
                'total_searches': num_searches,
                'success_rate': successful_searches / num_searches
            }
        else:
            return {
                'avg_search_time_ms': 0,
                'successful_searches': 0,
                'total_searches': num_searches,
                'success_rate': 0
            }
    
    def benchmark_concurrent_search(self, service_url: str, num_concurrent: int = 10, searches_per_thread: int = 10) -> Dict[str, Any]:
        """Benchmark concurrent search performance."""
        logger.info(f"Benchmarking concurrent search for {service_url} with {num_concurrent} threads")
        
        def search_worker():
            times = []
            for _ in range(searches_per_thread):
                query_embedding = np.random.rand(self.embedding_dim).astype(np.float32)
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
                
                start_time = time.time()
                try:
                    response = requests.post(
                        f"{service_url}/vectors/search",
                        json={
                            'embedding': query_embedding.tolist(),
                            'k': 5
                        },
                        timeout=30
                    )
                    if response.status_code == 200:
                        times.append((time.time() - start_time) * 1000)
                except:
                    pass
            return times
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(search_worker) for _ in range(num_concurrent)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        all_times = [time for result in results for time in result]
        
        if all_times:
            return {
                'total_time_s': total_time,
                'total_searches': len(all_times),
                'searches_per_second': len(all_times) / total_time,
                'avg_response_time_ms': statistics.mean(all_times),
                'p95_response_time_ms': np.percentile(all_times, 95)
            }
        else:
            return {
                'total_time_s': total_time,
                'total_searches': 0,
                'searches_per_second': 0
            }
    
    def get_service_stats(self, service_url: str) -> Dict[str, Any]:
        """Get service statistics."""
        try:
            response = requests.get(f"{service_url}/stats", timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {}
    
    def run_full_benchmark(self, test_vectors: int = 1000) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        logger.info(f"Starting full benchmark with {test_vectors} test vectors")
        
        # Generate test data
        test_data = self.generate_test_data(test_vectors)
        
        results = {}
        
        for service_name, service_url in self.services.items():
            logger.info(f"Benchmarking {service_name}")
            
            if not self.wait_for_service(service_url):
                logger.warning(f"Skipping {service_name} - service not available")
                continue
            
            service_results = {
                'service_url': service_url,
                'test_vectors': test_vectors
            }
            
            try:
                # Reset service
                requests.post(f"{service_url}/admin/reset", timeout=30)
                time.sleep(2)
                
                # Benchmark insertion
                insert_results = self.benchmark_insert_performance(service_url, test_data)
                service_results['insert_performance'] = insert_results
                
                # Wait for indexing to complete
                time.sleep(5)
                
                # Get service stats
                stats = self.get_service_stats(service_url)
                service_results['service_stats'] = stats
                
                # Benchmark search
                search_results = self.benchmark_search_performance(service_url, 100)
                service_results['search_performance'] = search_results
                
                # Benchmark concurrent search
                concurrent_results = self.benchmark_concurrent_search(service_url, 5, 20)
                service_results['concurrent_performance'] = concurrent_results
                
                results[service_name] = service_results
                
            except Exception as e:
                logger.error(f"Benchmark failed for {service_name}: {e}")
                results[service_name] = {'error': str(e)}
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a performance comparison report."""
        report = []
        report.append("=" * 80)
        report.append("VECTOR DATABASE PERFORMANCE COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary table
        report.append("SEARCH PERFORMANCE SUMMARY:")
        report.append("-" * 50)
        report.append(f"{'Service':<15} {'Avg Search (ms)':<15} {'P95 (ms)':<10} {'Success Rate':<12}")
        report.append("-" * 50)
        
        for service_name, service_data in results.items():
            if 'error' not in service_data and 'search_performance' in service_data:
                search_perf = service_data['search_performance']
                avg_time = search_perf.get('avg_search_time_ms', 0)
                p95_time = search_perf.get('p95_search_time_ms', 0)
                success_rate = search_perf.get('success_rate', 0)
                
                report.append(f"{service_name:<15} {avg_time:<15.2f} {p95_time:<10.2f} {success_rate:<12.2%}")
        
        report.append("")
        report.append("DETAILED RESULTS:")
        report.append("-" * 50)
        
        for service_name, service_data in results.items():
            report.append(f"\n{service_name}:")
            if 'error' in service_data:
                report.append(f"  Error: {service_data['error']}")
                continue
            
            # Service stats
            if 'service_stats' in service_data:
                stats = service_data['service_stats']
                report.append(f"  Total Vectors: {stats.get('total_vectors', 'N/A')}")
                report.append(f"  GPU Enabled: {stats.get('gpu_enabled', 'N/A')}")
                report.append(f"  Database Type: {stats.get('index_type', stats.get('database_type', 'N/A'))}")
            
            # Insert performance
            if 'insert_performance' in service_data:
                insert_perf = service_data['insert_performance']['single_insert']
                report.append(f"  Single Insert Avg: {insert_perf.get('avg_time_ms', 0):.2f}ms")
            
            # Search performance
            if 'search_performance' in service_data:
                search_perf = service_data['search_performance']
                report.append(f"  Search Avg: {search_perf.get('avg_search_time_ms', 0):.2f}ms")
                report.append(f"  Search P95: {search_perf.get('p95_search_time_ms', 0):.2f}ms")
                report.append(f"  Success Rate: {search_perf.get('success_rate', 0):.2%}")
            
            # Concurrent performance
            if 'concurrent_performance' in service_data:
                concurrent_perf = service_data['concurrent_performance']
                report.append(f"  Concurrent QPS: {concurrent_perf.get('searches_per_second', 0):.2f}")
        
        return "\n".join(report)


def main():
    """Run the benchmark suite."""
    benchmark = VectorDBBenchmark()
    
    # Run benchmarks with different dataset sizes
    test_sizes = [1000, 5000, 10000]
    
    for test_size in test_sizes:
        logger.info(f"Running benchmark with {test_size} vectors")
        results = benchmark.run_full_benchmark(test_size)
        
        # Save results
        results_file = f"/app/data/benchmark_results_{test_size}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate and save report
        report = benchmark.generate_report(results)
        report_file = f"/app/data/benchmark_report_{test_size}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\nBenchmark Results for {test_size} vectors:")
        print(report)
        print(f"\nResults saved to: {results_file}")
        print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()
