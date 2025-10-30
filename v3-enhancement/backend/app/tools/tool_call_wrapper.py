"""
Tool call wrapper with retry logic, circuit breakers, and telemetry.
Provides standardized observability and resilience for all tool operations.

Phase 3: Production-ready wrapper for tool execution with comprehensive monitoring.
"""
import asyncio
import logging
import time
import functools
from typing import Any, Callable, Optional, Dict, TypeVar, Coroutine
from contextlib import asynccontextmanager
from enum import Enum

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryCallState,
    before_sleep_log
)
from pybreaker import CircuitBreaker, CircuitBreakerError

from .base_tool import ToolResult, ToolStatus

logger = logging.getLogger(__name__)

# Type variable for generic function wrapping
T = TypeVar('T')


class ToolCallError(Exception):
    """Base exception for tool call errors."""
    pass


class ToolTimeoutError(ToolCallError):
    """Tool execution timeout error."""
    pass


class ToolCircuitBreakerError(ToolCallError):
    """Circuit breaker open error."""
    pass


# ===========================
# OpenTelemetry Integration
# ===========================

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    OTEL_AVAILABLE = True
    tracer = trace.get_tracer(__name__)
except ImportError:
    OTEL_AVAILABLE = False
    tracer = None
    logger.warning("OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk")


class SpanContext:
    """
    Fallback span context for when OpenTelemetry is not available.
    Provides no-op interface matching OTel API.
    """
    def __init__(self, name: str):
        self.name = name
        self.attributes = {}
    
    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value
    
    def set_status(self, status: Any) -> None:
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def get_span(name: str):
    """
    Get OpenTelemetry span or fallback.
    
    Args:
        name: Span name
        
    Returns:
        Span context
    """
    if OTEL_AVAILABLE and tracer:
        return tracer.start_as_current_span(name)
    else:
        return SpanContext(name)


# ===========================
# Circuit Breaker Configuration
# ===========================

class CircuitBreakerConfig:
    """Configuration for circuit breakers."""
    
    def __init__(
        self,
        fail_max: int = 5,
        timeout: int = 60,
        expected_exception: type = Exception,
        name: Optional[str] = None
    ):
        """
        Initialize circuit breaker configuration.
        
        Args:
            fail_max: Maximum failures before opening circuit
            timeout: Seconds before attempting to close circuit
            expected_exception: Exception type to track
            name: Circuit breaker name
        """
        self.fail_max = fail_max
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.name = name or "default"


# Global circuit breakers per tool
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(tool_name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """
    Get or create circuit breaker for a tool.
    
    Args:
        tool_name: Tool identifier
        config: Circuit breaker configuration
        
    Returns:
        Circuit breaker instance
    """
    if tool_name not in _circuit_breakers:
        if config is None:
            config = CircuitBreakerConfig(name=tool_name)
        
        _circuit_breakers[tool_name] = CircuitBreaker(
            fail_max=config.fail_max,
            timeout_duration=config.timeout,
            expected_exception=config.expected_exception,
            name=config.name
        )
        
        logger.info(
            f"Created circuit breaker for '{tool_name}': "
            f"fail_max={config.fail_max}, timeout={config.timeout}s"
        )
    
    return _circuit_breakers[tool_name]


def reset_circuit_breaker(tool_name: str) -> None:
    """
    Reset circuit breaker for a tool.
    
    Args:
        tool_name: Tool identifier
    """
    if tool_name in _circuit_breakers:
        _circuit_breakers[tool_name].call(lambda: None)
        logger.info(f"Reset circuit breaker for '{tool_name}'")


# ===========================
# Retry Configuration
# ===========================

class RetryConfig:
    """Configuration for retry logic."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        wait_multiplier: float = 1.0,
        wait_min: float = 1.0,
        wait_max: float = 10.0,
        retry_exceptions: tuple = (Exception,)
    ):
        """
        Initialize retry configuration.
        
        Args:
            max_attempts: Maximum retry attempts
            wait_multiplier: Exponential backoff multiplier
            wait_min: Minimum wait time between retries (seconds)
            wait_max: Maximum wait time between retries (seconds)
            retry_exceptions: Exception types to retry on
        """
        self.max_attempts = max_attempts
        self.wait_multiplier = wait_multiplier
        self.wait_min = wait_min
        self.wait_max = wait_max
        self.retry_exceptions = retry_exceptions


def create_retry_decorator(config: RetryConfig, tool_name: str):
    """
    Create retry decorator with configuration.
    
    Args:
        config: Retry configuration
        tool_name: Tool name for logging
        
    Returns:
        Configured retry decorator
    """
    return retry(
        stop=stop_after_attempt(config.max_attempts),
        wait=wait_exponential(
            multiplier=config.wait_multiplier,
            min=config.wait_min,
            max=config.wait_max
        ),
        retry=retry_if_exception_type(config.retry_exceptions),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )


# ===========================
# Tool Call Wrapper Context Manager
# ===========================

@asynccontextmanager
async def tool_call_context(
    tool_name: str,
    operation: str,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **metadata
):
    """
    Context manager for tool calls with telemetry and logging.
    
    Args:
        tool_name: Tool identifier
        operation: Operation name (e.g., 'search', 'store_memory')
        request_id: Request correlation ID
        session_id: Session identifier
        **metadata: Additional metadata to log
        
    Yields:
        Span context for setting attributes
        
    Example:
        async with tool_call_context('rag', 'search', request_id='123') as span:
            span.set_attribute('query', 'test')
            result = await rag_tool.search('test')
    """
    span_name = f"tool.{tool_name}.{operation}"
    start_time = time.time()
    
    # Create structured log context
    log_context = {
        "tool_name": tool_name,
        "operation": operation,
        "request_id": request_id,
        "session_id": session_id,
        **metadata
    }
    
    logger.info(
        f"Tool call started: {tool_name}.{operation}",
        extra=log_context
    )
    
    # Start OpenTelemetry span
    with get_span(span_name) as span:
        # Set span attributes
        if hasattr(span, 'set_attribute'):
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("tool.operation", operation)
            
            if request_id:
                span.set_attribute("request.id", request_id)
            if session_id:
                span.set_attribute("session.id", session_id)
            
            for key, value in metadata.items():
                if value is not None:
                    span.set_attribute(f"tool.{key}", str(value))
        
        try:
            yield span
            
            # Success logging
            duration = time.time() - start_time
            logger.info(
                f"Tool call completed: {tool_name}.{operation} "
                f"(duration: {duration:.3f}s)",
                extra={**log_context, "duration_seconds": duration, "status": "success"}
            )
            
            if hasattr(span, 'set_status') and OTEL_AVAILABLE:
                span.set_status(Status(StatusCode.OK))
            
        except Exception as e:
            # Error logging
            duration = time.time() - start_time
            logger.error(
                f"Tool call failed: {tool_name}.{operation} "
                f"(duration: {duration:.3f}s, error: {e})",
                extra={
                    **log_context,
                    "duration_seconds": duration,
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                exc_info=True
            )
            
            if hasattr(span, 'set_status') and OTEL_AVAILABLE:
                span.set_status(
                    Status(StatusCode.ERROR, description=str(e))
                )
            
            raise


# ===========================
# Tool Call Wrapper Decorator
# ===========================

def with_tool_call_wrapper(
    tool_name: str,
    operation: Optional[str] = None,
    retry_config: Optional[RetryConfig] = None,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    timeout: Optional[float] = None,
    convert_to_tool_result: bool = True
):
    """
    Decorator for wrapping tool calls with retry, circuit breaker, and telemetry.
    
    Args:
        tool_name: Tool identifier
        operation: Operation name (default: function name)
        retry_config: Retry configuration
        circuit_breaker_config: Circuit breaker configuration
        timeout: Execution timeout in seconds
        convert_to_tool_result: Convert exceptions to ToolResult
        
    Returns:
        Decorated function
        
    Example:
        @with_tool_call_wrapper('rag', 'search', timeout=30.0)
        async def search_wrapper(query: str, **kwargs):
            return await rag_tool.search(query, **kwargs)
    """
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        # Determine operation name
        op_name = operation or func.__name__
        
        # Create retry decorator if config provided
        retry_decorator = None
        if retry_config:
            retry_decorator = create_retry_decorator(retry_config, tool_name)
        
        # Get circuit breaker
        circuit_breaker = get_circuit_breaker(tool_name, circuit_breaker_config)
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract context from kwargs
            request_id = kwargs.pop('request_id', None)
            session_id = kwargs.pop('session_id', None)
            
            # Execute with telemetry
            async with tool_call_context(
                tool_name=tool_name,
                operation=op_name,
                request_id=request_id,
                session_id=session_id
            ) as span:
                try:
                    # Check circuit breaker
                    if circuit_breaker.current_state == 'open':
                        raise ToolCircuitBreakerError(
                            f"Circuit breaker open for tool '{tool_name}'"
                        )
                    
                    # Define execution function
                    async def execute():
                        # Apply timeout if specified
                        if timeout:
                            try:
                                return await asyncio.wait_for(
                                    func(*args, **kwargs),
                                    timeout=timeout
                                )
                            except asyncio.TimeoutError:
                                raise ToolTimeoutError(
                                    f"Tool '{tool_name}' operation '{op_name}' "
                                    f"timed out after {timeout}s"
                                )
                        else:
                            return await func(*args, **kwargs)
                    
                    # Apply retry if configured
                    if retry_decorator:
                        execute = retry_decorator(execute)
                    
                    # Execute through circuit breaker
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        circuit_breaker.call,
                        lambda: asyncio.run(execute())
                    )
                    
                    # Record success in span
                    if hasattr(span, 'set_attribute'):
                        span.set_attribute("tool.success", True)
                    
                    return result
                    
                except CircuitBreakerError as e:
                    # Circuit breaker is open
                    logger.warning(
                        f"Circuit breaker open for '{tool_name}': {e}",
                        extra={
                            "tool_name": tool_name,
                            "operation": op_name,
                            "circuit_breaker_state": circuit_breaker.current_state
                        }
                    )
                    
                    if convert_to_tool_result:
                        return ToolResult.error_result(
                            error=f"Service temporarily unavailable: {tool_name}",
                            metadata={
                                "tool": tool_name,
                                "operation": op_name,
                                "circuit_breaker_open": True
                            }
                        )
                    else:
                        raise ToolCircuitBreakerError(str(e))
                
                except Exception as e:
                    # Record error in span
                    if hasattr(span, 'set_attribute'):
                        span.set_attribute("tool.success", False)
                        span.set_attribute("tool.error_type", type(e).__name__)
                    
                    if convert_to_tool_result:
                        return ToolResult.error_result(
                            error=str(e),
                            metadata={
                                "tool": tool_name,
                                "operation": op_name,
                                "error_type": type(e).__name__
                            }
                        )
                    else:
                        raise
        
        return wrapper
    
    return decorator


# ===========================
# Convenience Functions
# ===========================

async def call_tool_with_wrapper(
    tool,
    method_name: str,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    retry_config: Optional[RetryConfig] = None,
    timeout: Optional[float] = None,
    **kwargs
) -> ToolResult:
    """
    Call a tool method with automatic wrapping.
    
    Args:
        tool: Tool instance
        method_name: Method name to call
        request_id: Request correlation ID
        session_id: Session identifier
        retry_config: Retry configuration
        timeout: Execution timeout
        **kwargs: Arguments to pass to tool method
        
    Returns:
        ToolResult from tool execution
        
    Example:
        result = await call_tool_with_wrapper(
            rag_tool,
            'search',
            request_id='123',
            query='test',
            timeout=30.0
        )
    """
    tool_name = getattr(tool, 'name', tool.__class__.__name__)
    
    # Get method
    method = getattr(tool, method_name)
    if not callable(method):
        raise AttributeError(f"Tool '{tool_name}' has no callable method '{method_name}'")
    
    # Wrap method
    wrapped_method = with_tool_call_wrapper(
        tool_name=tool_name,
        operation=method_name,
        retry_config=retry_config,
        timeout=timeout,
        convert_to_tool_result=True
    )(method)
    
    # Execute
    return await wrapped_method(
        **kwargs,
        request_id=request_id,
        session_id=session_id
    )


def get_tool_metrics(tool_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get metrics for circuit breakers.
    
    Args:
        tool_name: Specific tool name, or None for all tools
        
    Returns:
        Dictionary with circuit breaker metrics
    """
    if tool_name:
        if tool_name in _circuit_breakers:
            cb = _circuit_breakers[tool_name]
            return {
                "tool": tool_name,
                "state": cb.current_state,
                "fail_counter": cb.fail_counter,
                "last_failure": cb.last_failure
            }
        else:
            return {"tool": tool_name, "error": "No circuit breaker found"}
    else:
        return {
            name: {
                "state": cb.current_state,
                "fail_counter": cb.fail_counter,
                "last_failure": cb.last_failure
            }
            for name, cb in _circuit_breakers.items()
        }


# Export public API
__all__ = [
    'tool_call_context',
    'with_tool_call_wrapper',
    'call_tool_with_wrapper',
    'RetryConfig',
    'CircuitBreakerConfig',
    'get_circuit_breaker',
    'reset_circuit_breaker',
    'get_tool_metrics',
    'ToolCallError',
    'ToolTimeoutError',
    'ToolCircuitBreakerError'
]
