"""Async optimization utilities for high-performance API responses."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable, Awaitable
from functools import wraps
import time
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


class AsyncBatchProcessor:
    """Batch processor for optimizing multiple async operations."""
    
    def __init__(self, max_batch_size: int = 16, max_wait_time: float = 0.01):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self._pending_requests = []
        self._batch_lock = asyncio.Lock()
        self._processing = False
    
    async def add_to_batch(self, operation: Callable, *args, **kwargs) -> Any:
        """Add operation to batch for processing."""
        future = asyncio.Future()
        request = {
            'operation': operation,
            'args': args,
            'kwargs': kwargs,
            'future': future,
            'timestamp': time.time()
        }
        
        async with self._batch_lock:
            self._pending_requests.append(request)
            
            if len(self._pending_requests) >= self.max_batch_size or not self._processing:
                asyncio.create_task(self._process_batch())
        
        return await future
    
    async def _process_batch(self):
        """Process pending requests in batch."""
        if self._processing:
            return
        
        self._processing = True
        
        try:
            while self._pending_requests:
                async with self._batch_lock:
                    current_batch = self._pending_requests[:self.max_batch_size]
                    self._pending_requests = self._pending_requests[self.max_batch_size:]
                
                if current_batch:
                    await self._execute_batch(current_batch)
                
                # Small delay to allow more requests to accumulate
                if self._pending_requests:
                    await asyncio.sleep(0.001)
        
        finally:
            self._processing = False
    
    async def _execute_batch(self, batch: List[Dict[str, Any]]):
        """Execute a batch of operations concurrently."""
        tasks = []
        
        for request in batch:
            task = asyncio.create_task(
                self._execute_single_request(request)
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_single_request(self, request: Dict[str, Any]):
        """Execute a single request and set its future result."""
        try:
            operation = request['operation']
            args = request['args']
            kwargs = request['kwargs']
            
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
            
            request['future'].set_result(result)
            
        except Exception as e:
            request['future'].set_exception(e)


class ConnectionPool:
    """Optimized connection pool for external services."""
    
    def __init__(self, max_connections: int = 100, timeout: float = 5.0):
        self.max_connections = max_connections
        self.timeout = timeout
        self._connections = asyncio.Queue(maxsize=max_connections)
        self._created_connections = 0
        self._lock = asyncio.Lock()
    
    async def get_connection(self):
        """Get a connection from the pool."""
        try:
            return await asyncio.wait_for(
                self._connections.get(), 
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            async with self._lock:
                if self._created_connections < self.max_connections:
                    self._created_connections += 1
                    return await self._create_connection()
                else:
                    raise Exception("Connection pool exhausted")
    
    async def return_connection(self, connection):
        """Return a connection to the pool."""
        try:
            await self._connections.put(connection)
        except asyncio.QueueFull:
            await self._close_connection(connection)
    
    async def _create_connection(self):
        """Create a new connection."""
        import aiohttp
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            connector=aiohttp.TCPConnector(
                limit=self.max_connections,
                enable_cleanup_closed=True,
                force_close=True
            )
        )
    
    async def _close_connection(self, connection):
        """Close a connection."""
        try:
            await connection.close()
        except Exception:
            pass


class CacheManager:
    """High-performance async cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = {}
        self._access_times = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() < entry['expires']:
                    self._access_times[key] = time.time()
                    return entry['value']
                else:
                    del self._cache[key]
                    del self._access_times[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache."""
        if ttl is None:
            ttl = self.default_ttl
        
        async with self._lock:
            # Evict if cache is full
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_lru()
            
            self._cache[key] = {
                'value': value,
                'expires': time.time() + ttl
            }
            self._access_times[key] = time.time()
    
    async def _evict_lru(self):
        """Evict least recently used item."""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.keys(), 
                     key=lambda k: self._access_times[k])
        del self._cache[lru_key]
        del self._access_times[lru_key]
    
    async def clear(self):
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()


def async_cache(ttl: float = 300, max_size: int = 1000):
    """Decorator for caching async function results."""
    cache = CacheManager(max_size=max_size, default_ttl=ttl)
    
    def decorator(func: Callable[..., Awaitable[Any]]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl)
            return result
        
        wrapper._cache = cache  # Expose cache for manual management
        return wrapper
    
    return decorator


class ParallelExecutor:
    """Execute multiple async operations in parallel with optimization."""
    
    def __init__(self, max_concurrency: int = 50):
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)
    
    async def execute_parallel(self, operations: List[Callable[..., Awaitable[Any]]], 
                             *args_list, **kwargs_list) -> List[Any]:
        """Execute operations in parallel with concurrency control."""
        async def bounded_operation(op, args, kwargs):
            async with self._semaphore:
                return await op(*args, **kwargs)
        
        tasks = []
        for i, operation in enumerate(operations):
            args = args_list[i] if i < len(args_list) else ()
            kwargs = kwargs_list.get(i, {})
            task = asyncio.create_task(bounded_operation(operation, args, kwargs))
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def execute_with_timeout(self, operation: Callable[..., Awaitable[Any]], 
                                 timeout: float, *args, **kwargs) -> Any:
        """Execute operation with timeout."""
        async with self._semaphore:
            return await asyncio.wait_for(operation(*args, **kwargs), timeout=timeout)


# Global instances
batch_processor = AsyncBatchProcessor()
connection_pool = ConnectionPool()
parallel_executor = ParallelExecutor()


def optimize_async_endpoint(cache_ttl: float = 60, max_concurrency: int = 10):
    """Decorator to optimize async endpoints with caching and concurrency control."""
    def decorator(func: Callable[..., Awaitable[Any]]):
        cached_func = async_cache(ttl=cache_ttl)(func)
        semaphore = asyncio.Semaphore(max_concurrency)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with semaphore:
                return await cached_func(*args, **kwargs)
        
        return wrapper
    
    return decorator
