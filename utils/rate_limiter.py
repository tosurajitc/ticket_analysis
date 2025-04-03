import time
import random
from typing import Callable, Any, Optional, Dict
import json

class RateLimiter:
    def __init__(self, base_delay: float = 2.0, max_retries: int = 3, max_delay: float = 120.0):
        self.base_delay = base_delay
        self.max_retries = max_retries
        self.max_delay = max_delay
        self.last_request_time = 0
    
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        last_exception = None
        
        # Enforce minimum time between requests (3 seconds)
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < 3 and self.last_request_time > 0:
            sleep_time = 3 - time_since_last
            print(f"Rate limiting: Waiting {sleep_time:.2f} seconds between requests...")
            time.sleep(sleep_time)
        
        for attempt in range(self.max_retries + 1):
            try:
                self.last_request_time = time.time()
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                # Check if this is a rate limit error
                is_rate_limit = self._is_rate_limit_error(e)
                wait_time = self._extract_wait_time(e) if is_rate_limit else None
                
                if attempt == self.max_retries:
                    # Last attempt failed, re-raise the exception
                    raise last_exception
                
                # Calculate delay with exponential backoff and jitter
                if wait_time:
                    # Use the wait time from the rate limit error
                    delay = wait_time + 5  # Add buffer to the recommended wait time
                else:
                    # Use exponential backoff with jitter
                    delay = min(
                        self.max_delay,
                        self.base_delay * (2 ** attempt) * (0.5 + random.uniform(0, 0.5))
                    )
                
                print(f"Attempt {attempt+1} failed: {str(e)}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                self.last_request_time = time.time()  # Update after sleep
    
    def _is_rate_limit_error(self, exception: Exception) -> bool:
        """
        Check if an exception is due to rate limiting.
        """
        err_str = str(exception).lower()
        return (
            "rate limit" in err_str or 
            "429" in err_str or 
            "too many requests" in err_str
        )
    
    def _extract_wait_time(self, exception: Exception) -> Optional[float]:
        """
        Extract the wait time from a rate limit error message.
        """
        err_str = str(exception)
        
        # Try to extract wait time from GROQ-specific format
        try:
            if "Please try again in" in err_str:
                time_part = err_str.split("Please try again in")[1].split(".")[0].strip()
                
                # Parse time string like "1m42s"
                minutes = 0
                seconds = 0
                
                if "m" in time_part:
                    minutes_part = time_part.split("m")[0]
                    minutes = int(minutes_part)
                    time_part = time_part.split("m")[1]
                
                if "s" in time_part:
                    seconds_part = time_part.split("s")[0]
                    seconds = float(seconds_part)
                
                return minutes * 60 + seconds
        except:
            pass
        
        # Check for error code in json format
        try:
            if "Error code: 429" in err_str and "{" in err_str:
                json_str = err_str.split("{", 1)[1].rsplit("}", 1)[0]
                json_str = "{" + json_str + "}"
                
                error_data = json.loads(json_str)
                if "error" in error_data and "message" in error_data["error"]:
                    message = error_data["error"]["message"]
                    if "try again in" in message.lower():
                        # Extract time part (e.g., "1m42s")
                        time_parts = message.lower().split("try again in")[1].split(".")[0].strip()
                        
                        # Parse time string like "1m42s"
                        minutes = 0
                        seconds = 0
                        
                        if "m" in time_parts:
                            minutes_part = time_parts.split("m")[0]
                            minutes = int(minutes_part)
                            time_parts = time_parts.split("m")[1]
                        
                        if "s" in time_parts:
                            seconds_part = time_parts.split("s")[0]
                            seconds = float(seconds_part)
                        
                        return minutes * 60 + seconds
        except:
            pass
        
        # Default backoff time
        return 5.0