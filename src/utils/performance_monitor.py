"""Performance monitoring utilities for API optimization."""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List
from functools import wraps
from contextlib import asynccontextmanager
import psutil
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Real-time performance monitoring for API endpoints and hybrid vector store."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: {
            'response_times': deque(maxlen=max_history),
            'memory_usage': deque(maxlen=max_history),
            'gpu_usage': deque(maxlen=max_history),
            'error_count': 0,
            'total_requests': 0,
            'concurrent_requests': 0,
            'faiss_hits': 0,
            'qdrant_hits': 0,
            'hybrid_searches': 0,
            'cache_efficiency': deque(maxlen=max_history)
        })
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._hybrid_stats = {
            'total_faiss_hits': 0,
            'total_qdrant_hits': 0,
            'total_hybrid_searches': 0,
            'avg_faiss_time': 0.0,
            'avg_qdrant_time': 0.0,
            'cache_hit_rate': 0.0
        }
    
    def record_request(self, endpoint: str, response_time: float, 
                      memory_mb: float, gpu_mb: Optional[float] = None,
                      error: bool = False, search_source: Optional[str] = None):
        """Record performance metrics for a request."""
        with self._lock:
            metrics = self.metrics[endpoint]
            metrics['response_times'].append(response_time)
            metrics['memory_usage'].append(memory_mb)
            if gpu_mb is not None:
                metrics['gpu_usage'].append(gpu_mb)
            metrics['total_requests'] += 1
            if error:
                metrics['error_count'] += 1
            
            # Track hybrid vector store performance
            if search_source:
                if search_source == 'faiss':
                    metrics['faiss_hits'] += 1
                    self._hybrid_stats['total_faiss_hits'] += 1
                elif search_source == 'qdrant':
                    metrics['qdrant_hits'] += 1
                    self._hybrid_stats['total_qdrant_hits'] += 1
                elif search_source == 'hybrid':
                    metrics['hybrid_searches'] += 1
                    self._hybrid_stats['total_hybrid_searches'] += 1
                
                # Update cache efficiency
                total_searches = (self._hybrid_stats['total_faiss_hits'] + 
                                self._hybrid_stats['total_qdrant_hits'] + 
                                self._hybrid_stats['total_hybrid_searches'])
                if total_searches > 0:
                    cache_rate = self._hybrid_stats['total_faiss_hits'] / total_searches
                    metrics['cache_efficiency'].append(cache_rate)
                    self._hybrid_stats['cache_hit_rate'] = cache_rate
    
    def get_stats(self, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            if endpoint:
                return self._calculate_endpoint_stats(endpoint)
            
            # Global stats
            all_stats = {}
            for ep in self.metrics:
                all_stats[ep] = self._calculate_endpoint_stats(ep)
            
            return {
                'endpoints': all_stats,
                'hybrid_vector_store': self._hybrid_stats.copy(),
                'uptime_seconds': time.time() - self._start_time,
                'system_memory_percent': psutil.virtual_memory().percent,
                'system_cpu_percent': psutil.cpu_percent()
            }
    
    def _calculate_endpoint_stats(self, endpoint: str) -> Dict[str, Any]:
        """Calculate statistics for a specific endpoint."""
        metrics = self.metrics[endpoint]
        response_times = list(metrics['response_times'])
        memory_usage = list(metrics['memory_usage'])
        cache_efficiency = list(metrics['cache_efficiency'])
        
        if not response_times:
            return {'no_data': True}
        
        stats = {
            'total_requests': metrics['total_requests'],
            'error_count': metrics['error_count'],
            'error_rate': metrics['error_count'] / max(metrics['total_requests'], 1),
            'concurrent_requests': metrics['concurrent_requests'],
            'response_time': {
                'avg': sum(response_times) / len(response_times),
                'min': min(response_times),
                'max': max(response_times),
                'p95': self._percentile(response_times, 95),
                'p99': self._percentile(response_times, 99)
            },
            'memory_usage': {
                'avg_mb': sum(memory_usage) / len(memory_usage),
                'max_mb': max(memory_usage),
                'current_mb': memory_usage[-1] if memory_usage else 0
            },
            'vector_store_performance': {
                'faiss_hits': metrics['faiss_hits'],
                'qdrant_hits': metrics['qdrant_hits'],
                'hybrid_searches': metrics['hybrid_searches'],
                'cache_efficiency': sum(cache_efficiency) / len(cache_efficiency) if cache_efficiency else 0.0
            }
        }
        
        return stats
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


# Global performance monitor
perf_monitor = PerformanceMonitor()


def monitor_performance(endpoint_name: str):
    """Decorator to monitor endpoint performance."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                gpu_memory = None
                error = False
                
                try:
                    # Get GPU memory if available
                    try:
                        import torch
                        if torch.cuda.is_available():
                            gpu_memory = torch.cuda.memory_allocated(0) / 1024 / 1024
                    except ImportError:
                        pass
                    
                    perf_monitor.metrics[endpoint_name]['concurrent_requests'] += 1
                    result = await func(*args, **kwargs)
                    return result
                    
                except Exception as e:
                    error = True
                    raise
                    
                finally:
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    response_time = end_time - start_time
                    memory_used = end_memory - start_memory
                    
                    perf_monitor.record_request(
                        endpoint_name, response_time, memory_used, 
                        gpu_memory, error
                    )
                    perf_monitor.metrics[endpoint_name]['concurrent_requests'] -= 1
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                gpu_memory = None
                error = False
                
                try:
                    # Get GPU memory if available
                    try:
                        import torch
                        if torch.cuda.is_available():
                            gpu_memory = torch.cuda.memory_allocated(0) / 1024 / 1024
                    except ImportError:
                        pass
                    
                    perf_monitor.metrics[endpoint_name]['concurrent_requests'] += 1
                    result = func(*args, **kwargs)
                    return result
                    
                except Exception as e:
                    error = True
                    raise
                    
                finally:
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    response_time = end_time - start_time
                    memory_used = end_memory - start_memory
                    
                    perf_monitor.record_request(
                        endpoint_name, response_time, memory_used, 
                        gpu_memory, error
                    )
                    perf_monitor.metrics[endpoint_name]['concurrent_requests'] -= 1
            
            return sync_wrapper
    
    return decorator


@asynccontextmanager
async def performance_context(operation_name: str):
    """Async context manager for performance monitoring."""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        perf_monitor.record_request(
            operation_name,
            end_time - start_time,
            end_memory - start_memory
        )


class GPUMemoryTracker:
    """Track GPU memory usage for optimization."""
    
    def __init__(self):
        self.peak_memory = 0
        self.current_memory = 0
        self._enabled = False
        
        try:
            import torch
            if torch.cuda.is_available():
                self._enabled = True
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
    
    def update(self):
        """Update GPU memory statistics."""
        if not self._enabled:
            return
        
        try:
            import torch
            self.current_memory = torch.cuda.memory_allocated(0) / 1024 / 1024
            self.peak_memory = torch.cuda.max_memory_allocated(0) / 1024 / 1024
        except Exception:
            pass
    
    def reset(self):
        """Reset peak memory statistics."""
        if not self._enabled:
            return
        
        try:
            import torch
            torch.cuda.reset_peak_memory_stats()
            self.peak_memory = 0
        except Exception:
            pass
    
    def get_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics."""
        self.update()
        return {
            'current_mb': self.current_memory,
            'peak_mb': self.peak_memory,
            'enabled': self._enabled
        }


# Global GPU memory tracker
gpu_tracker = GPUMemoryTracker()
