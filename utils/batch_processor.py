import time
from typing import List, Callable, TypeVar, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from .rate_limiter import RateLimiter

T = TypeVar('T')  # Generic type for inputs
R = TypeVar('R')  # Generic type for results

class BatchProcessor:
    """
    Process a batch of items with rate limiting and error handling.
    Supports parallel processing with configurable concurrency.
    """
    
    def __init__(self, concurrent_tasks: int = 1, rate_limiter: Optional[RateLimiter] = None):
        """
        Initialize the batch processor.
        
        Args:
            concurrent_tasks: Number of concurrent tasks to run
            rate_limiter: RateLimiter instance for handling rate limits
        """
        self.concurrent_tasks = concurrent_tasks
        self.rate_limiter = rate_limiter or RateLimiter()
    
    def process_batch(self, 
                      items: List[T], 
                      process_func: Callable[[T], R], 
                      fallback_func: Optional[Callable[[T, Exception], R]] = None) -> List[R]:
        """
        Process a batch of items with rate limiting.
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            fallback_func: Function to handle errors for each item
            
        Returns:
            List of results, in same order as input items
        """
        if not items:
            return []
        
        # Use single-threaded processing if concurrent_tasks is 1
        if self.concurrent_tasks == 1:
            return self._process_batch_sequential(items, process_func, fallback_func)
        else:
            return self._process_batch_parallel(items, process_func, fallback_func)
    
    def _process_batch_sequential(self, 
                                 items: List[T], 
                                 process_func: Callable[[T], R],
                                 fallback_func: Optional[Callable[[T, Exception], R]] = None) -> List[R]:
        """
        Process a batch of items sequentially with rate limiting.
        """
        results = []
        
        for item in items:
            try:
                # Execute the function with retry logic
                result = self.rate_limiter.execute_with_retry(process_func, item)
                results.append(result)
            except Exception as e:
                if fallback_func:
                    # Use fallback function if provided
                    fallback_result = fallback_func(item, e)
                    results.append(fallback_result)
                else:
                    # Use None as result if no fallback
                    results.append(None)
        
        return results
    
    def _process_batch_parallel(self, 
                               items: List[T], 
                               process_func: Callable[[T], R],
                               fallback_func: Optional[Callable[[T, Exception], R]] = None) -> List[R]:
        """
        Process a batch of items in parallel with rate limiting.
        """
        # Create a map to preserve the original order
        result_map = {i: None for i in range(len(items))}
        
        # Create a wrapper function that includes retry logic
        def wrapper(index: int, item: T) -> tuple[int, R]:
            try:
                result = self.rate_limiter.execute_with_retry(process_func, item)
                return index, result
            except Exception as e:
                if fallback_func:
                    # Use fallback function if provided
                    fallback_result = fallback_func(item, e)
                    return index, fallback_result
                else:
                    # Use None as result if no fallback
                    return index, None
        
        # Execute tasks in parallel with a thread pool
        with ThreadPoolExecutor(max_workers=self.concurrent_tasks) as executor:
            # Submit all tasks
            futures = [
                executor.submit(wrapper, i, item) 
                for i, item in enumerate(items)
            ]
            
            # Process results as they complete
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    result_map[idx] = result
                except Exception as e:
                    print(f"Unhandled exception in parallel task: {str(e)}")
        
        # Return results in original order
        return [result_map[i] for i in range(len(items))]