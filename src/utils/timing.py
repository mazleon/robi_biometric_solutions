"""Timing utilities for performance tracking."""

import time
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class TimingTracker:
    """Track timing for various operations."""
    
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}
    
    def start(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end(self, operation: str) -> float:
        """End timing an operation and return duration in milliseconds."""
        if operation not in self.start_times:
            logger.warning(f"No start time found for operation: {operation}")
            return 0.0
        
        duration_ms = (time.time() - self.start_times[operation]) * 1000
        self.timings[operation] = duration_ms
        del self.start_times[operation]
        
        logger.debug(f"Operation '{operation}' took {duration_ms:.2f}ms")
        return duration_ms
    
    def get_timing(self, operation: str) -> Optional[float]:
        """Get timing for a specific operation."""
        return self.timings.get(operation)
    
    def get_all_timings(self) -> Dict[str, float]:
        """Get all recorded timings."""
        return self.timings.copy()
    
    def reset(self) -> None:
        """Reset all timings."""
        self.timings.clear()
        self.start_times.clear()
    
    def log_summary(self, operation_name: str = "Operation", total_operation: Optional[str] = None) -> None:
        """Log a summary of all timings, with an optional total operation for accurate breakdown."""
        if not self.timings:
            logger.info(f"{operation_name} - No timing data available")
            return

        # Determine total time and components for breakdown
        if total_operation and total_operation in self.timings:
            total_time = self.timings[total_operation]
            components = {k: v for k, v in self.timings.items() if k != total_operation}
        else:
            # Fallback: sum all timings if no specific total is provided
            total_time = sum(self.timings.values())
            components = self.timings

        logger.info(f"{operation_name} timing summary:")
        logger.info(f"  Total time: {total_time:.2f}ms")

        # Calculate and show breakdown
        tracked_sum = sum(components.values())
        for op, duration in sorted(components.items(), key=lambda item: item[1], reverse=True):
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            logger.info(f"  - {op}: {duration:.2f}ms ({percentage:.1f}%)")

        # Show untracked time if significant
        if total_operation and total_time > tracked_sum:
            untracked_time = total_time - tracked_sum
            if untracked_time > 1.0:  # Only show if > 1ms
                percentage = (untracked_time / total_time) * 100 if total_time > 0 else 0
                logger.info(f"  - Other (untracked): {untracked_time:.2f}ms ({percentage:.1f}%)")

@contextmanager
def time_operation(operation_name: str, tracker: Optional[TimingTracker] = None):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000
        if tracker:
            tracker.timings[operation_name] = duration_ms
        logger.debug(f"Operation '{operation_name}' took {duration_ms:.2f}ms")


def log_timing(operation_name: str, start_time: float) -> float:
    """Log timing for an operation given start time."""
    duration_ms = (time.time() - start_time) * 1000
    logger.info(f"{operation_name} completed in {duration_ms:.2f}ms")
    return duration_ms
