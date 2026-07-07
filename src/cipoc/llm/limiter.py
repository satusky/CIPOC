import time
import random
from functools import wraps
from typing import Callable, Tuple, Type
import logging
from openai import RateLimitError

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class MaxRetriesExceededError(Exception):
    "Exception raised when max retry count is exceeded"
    def __init__(self, message: str):
        super().__init__(message)


class RateLimiter:
    def __init__(
        self,
        max_retries: int = 10,
        initial_delay: float = 1.0,
        exponential_base: float = 2.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        retry_on: Tuple[Type[Exception], ...] = (RateLimitError,),
        logger_name: str | None = None
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.exponential_base = exponential_base
        self.max_delay = max_delay
        self.jitter = jitter
        self.retry_on = retry_on
        self.logger = logging.getLogger(logger_name) if logger_name else logger

    def to_dict(self) -> dict:
        return {
            "max_retries": self.max_retries,
            "initial_delay": self.initial_delay,
            "exponential_base": self.exponential_base,
            "max_delay": self.max_delay,
            "jitter": self.jitter,
            "retry_on": [exc.__name__ for exc in self.retry_on],
            "logger_name": self.logger.name if self.logger else None,
        }
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            delay = self.initial_delay

            while attempts < self.max_retries:
                try:
                    return func(*args, **kwargs)
                
                except self.retry_on as e:
                    attempts += 1
                    if hasattr(e, "body") and isinstance(e.body, dict) and e.body.get("retry_after"):
                        delay = float(e.body["retry_after"])
                        self.logger.warning(
                            f"Attempt {attempts}/{self.max_retries} failed."
                            f"Retry-After: {delay:.2f}s..."
                        )
                    else:
                        if self.jitter:
                            delay *= self.exponential_base * (1 + random.random())
                        else:
                            delay *= self.exponential_base

                        delay = min(delay, self.max_delay)
                        self.logger.warning(
                            f"Attempt {attempts}/{self.max_retries} failed."
                            f"Retrying in {delay:.2f}s..."
                        )

                    time.sleep(delay)

            raise MaxRetriesExceededError(f"Max retries ({self.max_retries}) exceeded.")
        
        return wrapper
