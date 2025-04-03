import time
from typing import List, Callable, TypeVar, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from .rate_limiter import RateLimiter

T = TypeVar('T')  # Generic type for inputs
R = TypeVar('R')  # Generic type for results

class BatchProcessor:
    def __init__(self, concurrent_tasks: int = 1, rate_limiter: Optional[RateLimiter] = None):
        self.concurrent_tasks = concurrent_tasks
        self.rate_limiter = rate_limiter or RateLimiter()
    
    def process_batch(self, 
                      items: List[T], 
                      process_func: Callable[[T], R], 
                      fallback_func: Optional[Callable[[T, Exception], R]] = None) -> List[R]:
        if not items:
            return []
        
        if self.concurrent_tasks == 1:
            return self._process_batch_sequential(items, process_func, fallback_func)
        else:
            return self._process_batch_parallel(items, process_func, fallback_func)
    
    def _process_batch_sequential(self, 
                                 items: List[T], 
                                 process_func: Callable[[T], R],
                                 fallback_func: Optional[Callable[[T, Exception], R]] = None) -> List[R]:
        results = []
        
        for item in items:
            try:
                result = self.rate_limiter.execute_with_retry(process_func, item)
                results.append(result)
            except Exception as e:
                if fallback_func:
                    fallback_result = fallback_func(item, e)
                    results.append(fallback_result)
                else:
                    results.append(None)
        
        return results
    
    def _process_batch_parallel(self, 
                               items: List[T], 
                               process_func: Callable[[T], R],
                               fallback_func: Optional[Callable[[T, Exception], R]] = None) -> List[R]:
        result_map = {i: None for i in range(len(items))}
        
        def wrapper(index: int, item: T) -> tuple[int, R]:
            try:
                result = self.rate_limiter.execute_with_retry(process_func, item)
                return index, result
            except Exception as e:
                if fallback_func:
                    fallback_result = fallback_func(item, e)
                    return index, fallback_result
                else:
                    return index, None
        
        with ThreadPoolExecutor(max_workers=self.concurrent_tasks) as executor:
            futures = [
                executor.submit(wrapper, i, item) 
                for i, item in enumerate(items)
            ]
            
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    result_map[idx] = result
                except Exception as e:
                    print(f"Unhandled exception in parallel task: {str(e)}")
        
        return [result_map[i] for i in range(len(items))]