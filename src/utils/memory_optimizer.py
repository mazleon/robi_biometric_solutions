"""Memory optimization utilities for efficient resource management."""

import gc
import logging
import threading
import time
from typing import Dict, Any, Optional, List
from functools import wraps
import weakref
from collections import defaultdict
import psutil
import os

logger = logging.getLogger(__name__)


class MemoryManager:
    """Advanced memory management for face processing operations."""
    
    def __init__(self, gc_threshold: float = 0.8, cleanup_interval: int = 60):
        self.gc_threshold = gc_threshold  # Trigger cleanup at 80% memory usage
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
        self._memory_pools = defaultdict(list)
        self._weak_refs = set()
        self._lock = threading.Lock()
        
        # Start background cleanup thread
        self._cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self._cleanup_thread.start()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        return {
            'process_mb': memory_info.rss / 1024 / 1024,
            'process_percent': process.memory_percent(),
            'system_available_mb': virtual_memory.available / 1024 / 1024,
            'system_percent': virtual_memory.percent,
            'swap_percent': psutil.swap_memory().percent
        }
    
    def should_cleanup(self) -> bool:
        """Check if memory cleanup should be triggered."""
        memory_stats = self.get_memory_usage()
        return (
            memory_stats['system_percent'] > self.gc_threshold * 100 or
            time.time() - self.last_cleanup > self.cleanup_interval
        )
    
    def cleanup_memory(self, force: bool = False):
        """Perform memory cleanup operations."""
        if not force and not self.should_cleanup():
            return
        
        with self._lock:
            logger.debug("Starting memory cleanup...")
            
            # Clear memory pools
            for pool_name in list(self._memory_pools.keys()):
                self._memory_pools[pool_name].clear()
            
            # Clean up weak references
            dead_refs = [ref for ref in self._weak_refs if ref() is None]
            for ref in dead_refs:
                self._weak_refs.discard(ref)
            
            # Force garbage collection
            collected = gc.collect()
            
            # GPU memory cleanup if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
            
            self.last_cleanup = time.time()
            logger.debug(f"Memory cleanup completed. Collected {collected} objects.")
    
    def _background_cleanup(self):
        """Background thread for periodic memory cleanup."""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                if self.should_cleanup():
                    self.cleanup_memory()
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
    
    def register_for_cleanup(self, obj: Any, pool_name: str = "default"):
        """Register object for automatic cleanup."""
        with self._lock:
            if hasattr(obj, '__weakref__'):
                weak_ref = weakref.ref(obj)
                self._weak_refs.add(weak_ref)
            else:
                self._memory_pools[pool_name].append(obj)
    
    def get_pool_size(self, pool_name: str = "default") -> int:
        """Get size of memory pool."""
        with self._lock:
            return len(self._memory_pools[pool_name])


class ObjectPool:
    """Object pool for reusing expensive objects."""
    
    def __init__(self, factory_func, max_size: int = 10, cleanup_func=None):
        self.factory_func = factory_func
        self.cleanup_func = cleanup_func
        self.max_size = max_size
        self._pool = []
        self._lock = threading.Lock()
        self._created_count = 0
    
    def get_object(self):
        """Get object from pool or create new one."""
        with self._lock:
            if self._pool:
                return self._pool.pop()
            else:
                self._created_count += 1
                return self.factory_func()
    
    def return_object(self, obj):
        """Return object to pool."""
        with self._lock:
            if len(self._pool) < self.max_size:
                # Clean object if cleanup function provided
                if self.cleanup_func:
                    self.cleanup_func(obj)
                self._pool.append(obj)
            else:
                # Pool is full, discard object
                if self.cleanup_func:
                    self.cleanup_func(obj)
    
    def clear_pool(self):
        """Clear all objects from pool."""
        with self._lock:
            if self.cleanup_func:
                for obj in self._pool:
                    self.cleanup_func(obj)
            self._pool.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        with self._lock:
            return {
                'pool_size': len(self._pool),
                'max_size': self.max_size,
                'created_count': self._created_count
            }


def memory_efficient(cleanup_after: bool = True, pool_name: str = "default"):
    """Decorator for memory-efficient function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            memory_manager.register_for_cleanup(args, pool_name)
            memory_manager.register_for_cleanup(kwargs, pool_name)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if cleanup_after:
                    memory_manager.cleanup_memory()
        
        return wrapper
    return decorator


def gpu_memory_efficient(func):
    """Decorator for GPU memory-efficient operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import torch
            if torch.cuda.is_available():
                # Clear cache before operation
                torch.cuda.empty_cache()
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Clear cache after operation
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                return result
            else:
                return func(*args, **kwargs)
        except ImportError:
            return func(*args, **kwargs)
    
    return wrapper


class ImageBufferPool:
    """Specialized pool for image buffers to reduce allocation overhead."""
    
    def __init__(self, max_buffers: int = 20):
        self.max_buffers = max_buffers
        self._buffers = defaultdict(list)  # size -> [buffers]
        self._lock = threading.Lock()
    
    def get_buffer(self, size: tuple, dtype='uint8'):
        """Get buffer of specified size and type."""
        import numpy as np
        
        buffer_key = (size, dtype)
        
        with self._lock:
            if buffer_key in self._buffers and self._buffers[buffer_key]:
                buffer = self._buffers[buffer_key].pop()
                buffer.fill(0)  # Clear buffer
                return buffer
            else:
                return np.zeros(size, dtype=dtype)
    
    def return_buffer(self, buffer):
        """Return buffer to pool."""
        if buffer is None:
            return
        
        buffer_key = (buffer.shape, str(buffer.dtype))
        
        with self._lock:
            if len(self._buffers[buffer_key]) < self.max_buffers:
                self._buffers[buffer_key].append(buffer)
    
    def clear_all(self):
        """Clear all buffers from pool."""
        with self._lock:
            self._buffers.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer pool statistics."""
        with self._lock:
            stats = {}
            total_buffers = 0
            for key, buffers in self._buffers.items():
                size, dtype = key
                count = len(buffers)
                total_buffers += count
                stats[f"{size}_{dtype}"] = count
            
            stats['total_buffers'] = total_buffers
            return stats


# Global instances
memory_manager = MemoryManager()
image_buffer_pool = ImageBufferPool()


def create_numpy_pool():
    """Create object pool for numpy arrays."""
    def create_array():
        import numpy as np
        return np.empty((640, 640, 3), dtype=np.uint8)
    
    def cleanup_array(arr):
        if arr is not None:
            arr.fill(0)
    
    return ObjectPool(create_array, max_size=10, cleanup_func=cleanup_array)


def create_pil_pool():
    """Create object pool for PIL Images."""
    def create_image():
        from PIL import Image
        import numpy as np
        return Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
    
    def cleanup_image(img):
        if img is not None:
            img.close()
    
    return ObjectPool(create_image, max_size=5, cleanup_func=cleanup_image)


# Pre-created pools
numpy_pool = create_numpy_pool()
pil_pool = create_pil_pool()
