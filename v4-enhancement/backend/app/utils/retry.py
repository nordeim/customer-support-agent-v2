"""
Retry and resilience utilities for database and external service calls.
Implements retry logic, exponential backoff, and circuit breaker pattern.

Version: 1.0.0
"""
import asyncio
import logging
import time
import functools
from typing import Callable, Optional, Tuple, Type, Union, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RetryStrategy(str, Enum):
    """Retry strategy enumeration."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retry_on_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    retry_on_result: Optional[Callable[[Any], bool]] = None


class CircuitBreakerState(str, Enum):
    """Circuit breaker state."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failures exceeded threshold, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker implementation to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are rejected
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            success_threshold: Successes needed to close circuit from half-open
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_recovery():
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute async function through circuit breaker.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_recovery():
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.success_count = 0
                logger.info("Circuit breaker CLOSED after successful recovery")
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_count = 0
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(
                f"Circuit breaker OPENED after {self.failure_count} failures"
            )
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info("Circuit breaker manually reset to CLOSED")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


def calculate_retry_delay(
    attempt: int,
    config: RetryConfig
) -> float:
    """
    Calculate delay before next retry attempt.
    
    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration
        
    Returns:
        Delay in seconds
    """
    if config.strategy == RetryStrategy.FIXED:
        return config.initial_delay
    
    elif config.strategy == RetryStrategy.LINEAR:
        delay = config.initial_delay * (attempt + 1)
    
    elif config.strategy == RetryStrategy.EXPONENTIAL:
        delay = config.initial_delay * (config.exponential_base ** attempt)
    
    else:
        delay = config.initial_delay
    
    # Cap at max delay
    return min(delay, config.max_delay)


def retry(config: Optional[RetryConfig] = None):
    """
    Decorator for retrying function calls with configurable backoff.
    
    Args:
        config: Retry configuration
        
    Example:
        @retry(RetryConfig(max_attempts=3, strategy=RetryStrategy.EXPONENTIAL))
        def unreliable_function():
            ...
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    result = func(*args, **kwargs)
                    
                    # Check if result should trigger retry
                    if config.retry_on_result and config.retry_on_result(result):
                        if attempt < config.max_attempts - 1:
                            delay = calculate_retry_delay(attempt, config)
                            logger.warning(
                                f"{func.__name__} returned retry-worthy result, "
                                f"retrying in {delay:.2f}s (attempt {attempt + 1}/{config.max_attempts})"
                            )
                            time.sleep(delay)
                            continue
                    
                    return result
                    
                except config.retry_on_exceptions as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts - 1:
                        delay = calculate_retry_delay(attempt, config)
                        logger.warning(
                            f"{func.__name__} failed with {type(e).__name__}, "
                            f"retrying in {delay:.2f}s (attempt {attempt + 1}/{config.max_attempts}): {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {config.max_attempts} attempts: {e}"
                        )
            
            # All retries exhausted
            if last_exception:
                raise last_exception
            
            # This shouldn't happen, but just in case
            raise RuntimeError(f"{func.__name__} failed without exception")
        
        return wrapper
    return decorator


def async_retry(config: Optional[RetryConfig] = None):
    """
    Decorator for retrying async function calls with configurable backoff.
    
    Args:
        config: Retry configuration
        
    Example:
        @async_retry(RetryConfig(max_attempts=3))
        async def unreliable_async_function():
            ...
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    result = await func(*args, **kwargs)
                    
                    # Check if result should trigger retry
                    if config.retry_on_result and config.retry_on_result(result):
                        if attempt < config.max_attempts - 1:
                            delay = calculate_retry_delay(attempt, config)
                            logger.warning(
                                f"{func.__name__} returned retry-worthy result, "
                                f"retrying in {delay:.2f}s (attempt {attempt + 1}/{config.max_attempts})"
                            )
                            await asyncio.sleep(delay)
                            continue
                    
                    return result
                    
                except config.retry_on_exceptions as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts - 1:
                        delay = calculate_retry_delay(attempt, config)
                        logger.warning(
                            f"{func.__name__} failed with {type(e).__name__}, "
                            f"retrying in {delay:.2f}s (attempt {attempt + 1}/{config.max_attempts}): {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {config.max_attempts} attempts: {e}"
                        )
            
            # All retries exhausted
            if last_exception:
                raise last_exception
            
            raise RuntimeError(f"{func.__name__} failed without exception")
        
        return wrapper
    return decorator


__all__ = [
    'RetryConfig',
    'RetryStrategy',
    'CircuitBreaker',
    'CircuitBreakerState',
    'CircuitBreakerOpenError',
    'retry',
    'async_retry',
    'calculate_retry_delay'
]
