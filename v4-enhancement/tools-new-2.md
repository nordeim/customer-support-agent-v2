# 🚀 COMPLETE FILE IMPLEMENTATIONS (Continued)

## File 5: `backend/app/tools/tool_call_wrapper.py` (CRITICAL FIX - Complete Replacement)

```python
"""
Tool call wrapper with retry logic, circuit breakers, and telemetry.
Provides standardized observability and resilience for all tool operations.

Version: 2.0.0 (CRITICAL FIX: Async-compatible circuit breaker)
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
    before_sleep_log
)
from aiobreaker import CircuitBreaker, CircuitBreakerError

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
    logger.debug("OpenTelemetry not available - telemetry disabled")


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


def get_circuit_breaker(
    tool_name: str,
    config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """
    Get or create async circuit breaker for a tool.
    
    CRITICAL FIX: Uses aiobreaker instead of pybreaker for async compatibility.
    
    Args:
        tool_name: Tool identifier
        config: Circuit breaker configuration
        
    Returns:
        Async circuit breaker instance
    """
    if tool_name not in _circuit_breakers:
        if config is None:
            config = CircuitBreakerConfig(name=tool_name)
        
        # FIXED: Use aiobreaker for async compatibility
        _circuit_breakers[tool_name] = CircuitBreaker(
            fail_max=config.fail_max,
            timeout=config.timeout,
            expected_exception=config.expected_exception,
            name=config.name
        )
        
        logger.info(
            f"Created async circuit breaker for '{tool_name}': "
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
        cb = _circuit_breakers[tool_name]
        # aiobreaker doesn't have a direct reset, but we can recreate it
        _circuit_breakers.pop(tool_name)
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
    
    CRITICAL FIX v2.0.0: Now uses async-compatible circuit breaker (aiobreaker).
    
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
        
        # Get circuit breaker (async-compatible)
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
                    # Define execution function with timeout
                    async def execute_with_timeout():
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
                    
                    # Apply retry decorator if configured
                    if retry_decorator:
                        execute_with_timeout = retry_decorator(execute_with_timeout)
                    
                    # CRITICAL FIX: Execute through async circuit breaker
                    # Uses aiobreaker's async context manager
                    async with circuit_breaker:
                        result = await execute_with_timeout()
                    
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
                            "circuit_breaker_state": str(circuit_breaker.state)
                        }
                    )
                    
                    if convert_to_tool_result:
                        return ToolResult.error_result(
                            error=f"Service temporarily unavailable: {tool_name}",
                            metadata={
                                "tool": tool_name,
                                "operation": op_name,
                                "circuit_breaker_open": True,
                                "circuit_breaker_state": str(circuit_breaker.state)
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
        raise AttributeError(
            f"Tool '{tool_name}' has no callable method '{method_name}'"
        )
    
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
                "state": str(cb.state),
                "fail_counter": cb.fail_counter,
                "success_counter": cb.success_counter,
                "last_failure": str(cb.last_failure) if hasattr(cb, 'last_failure') else None
            }
        else:
            return {"tool": tool_name, "error": "No circuit breaker found"}
    else:
        return {
            name: {
                "state": str(cb.state),
                "fail_counter": cb.fail_counter,
                "success_counter": cb.success_counter
            }
            for name, cb in _circuit_breakers.items()
        }


def get_all_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """
    Get all circuit breaker instances.
    
    Returns:
        Dictionary mapping tool names to circuit breakers
    """
    return _circuit_breakers.copy()


async def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers."""
    for tool_name in list(_circuit_breakers.keys()):
        reset_circuit_breaker(tool_name)
    
    logger.info("All circuit breakers reset")


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
    'get_all_circuit_breakers',
    'reset_all_circuit_breakers',
    'ToolCallError',
    'ToolTimeoutError',
    'ToolCircuitBreakerError'
]
```

---

## File 6: `backend/app/tools/tool_adapters.py` (Thread-Safe - Complete Replacement)

```python
"""
Adapters for converting sync tool methods to async interface.
Provides backward compatibility during async migration.

Version: 2.0.0 (Thread-safe executor cleanup)
"""
import asyncio
import functools
import inspect
import threading
from typing import Any, Callable, Coroutine, TypeVar, cast
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# Thread pool for running sync functions in async context
_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()
_executor_shutdown = False


def get_executor() -> ThreadPoolExecutor:
    """
    Get thread pool executor with thread-safe lazy initialization.
    
    FIXED: Thread-safe singleton pattern with double-checked locking.
    
    Returns:
        ThreadPoolExecutor instance
    """
    global _executor, _executor_shutdown
    
    # Fast path: executor already exists
    if _executor is not None and not _executor_shutdown:
        return _executor
    
    # Slow path: need to create executor
    with _executor_lock:
        # Double-check inside lock
        if _executor is None or _executor_shutdown:
            if _executor_shutdown:
                logger.warning("Attempting to use executor after shutdown, recreating...")
            
            _executor = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="tool_adapter_"
            )
            _executor_shutdown = False
            logger.info("Created thread pool executor for tool adapters")
    
    return _executor


def sync_to_async_adapter(func: Callable[..., T]) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Convert a synchronous function to async using thread pool executor.
    
    Args:
        func: Synchronous function to wrap
        
    Returns:
        Async function that runs sync func in thread pool
        
    Example:
        def sync_search(query: str) -> dict:
            return {"results": query}
        
        async_search = sync_to_async_adapter(sync_search)
        result = await async_search("test")
    """
    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_event_loop()
        executor = get_executor()
        
        return await loop.run_in_executor(
            executor,
            functools.partial(func, *args, **kwargs)
        )
    
    return async_wrapper


def ensure_async(func: F) -> F:
    """
    Decorator to ensure a function is async.
    If function is sync, converts it using sync_to_async_adapter.
    If already async, returns as-is.
    
    Args:
        func: Function to ensure is async
        
    Returns:
        Async version of function
        
    Example:
        @ensure_async
        def my_function(x: int) -> int:
            return x * 2
        
        # Can now be called with await
        result = await my_function(5)
    """
    if inspect.iscoroutinefunction(func):
        # Already async, return as-is
        return func
    
    # Wrap sync function
    async_func = sync_to_async_adapter(func)
    return cast(F, async_func)


class AsyncToolAdapter:
    """
    Adapter to wrap a legacy sync tool and provide async interface.
    
    Example:
        legacy_tool = OldSyncTool()
        async_tool = AsyncToolAdapter(legacy_tool)
        
        # Now can use async methods
        result = await async_tool.execute(query="test")
    """
    
    def __init__(self, tool: Any):
        """
        Wrap a sync tool with async interface.
        
        Args:
            tool: Legacy tool instance to wrap
        """
        self._tool = tool
        self.name = getattr(tool, 'name', 'unknown')
        self.description = getattr(tool, 'description', '')
        self.initialized = getattr(tool, 'initialized', False)
        
        logger.info(f"Created async adapter for tool '{self.name}'")
    
    async def initialize(self) -> None:
        """Initialize wrapped tool."""
        if hasattr(self._tool, 'initialize'):
            # Tool already has async initialize
            await self._tool.initialize()
        elif hasattr(self._tool, '_initialize'):
            # Legacy sync initialize
            init_func = sync_to_async_adapter(self._tool._initialize)
            await init_func()
        
        self.initialized = True
    
    async def execute(self, **kwargs: Any) -> Any:
        """
        Execute wrapped tool method.
        
        Attempts to call in order:
        1. async execute() if exists
        2. sync execute() wrapped in adapter
        3. __call__() method wrapped in adapter
        """
        if hasattr(self._tool, 'execute'):
            execute_fn = self._tool.execute
            if inspect.iscoroutinefunction(execute_fn):
                return await execute_fn(**kwargs)
            else:
                async_execute = sync_to_async_adapter(execute_fn)
                return await async_execute(**kwargs)
        
        elif callable(self._tool):
            if inspect.iscoroutinefunction(self._tool.__call__):
                return await self._tool(**kwargs)
            else:
                async_call = sync_to_async_adapter(self._tool.__call__)
                return await async_call(**kwargs)
        
        else:
            raise NotImplementedError(
                f"Tool '{self.name}' has no execute or __call__ method"
            )
    
    async def cleanup(self) -> None:
        """Cleanup wrapped tool."""
        if hasattr(self._tool, 'cleanup'):
            cleanup_fn = self._tool.cleanup
            if inspect.iscoroutinefunction(cleanup_fn):
                await cleanup_fn()
            else:
                async_cleanup = sync_to_async_adapter(cleanup_fn)
                await async_cleanup()
    
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped tool."""
        return getattr(self._tool, name)


def ensure_tool_async(tool: Any) -> Any:
    """
    Ensure a tool has async interface.
    If tool is already async-compatible, returns as-is.
    Otherwise, wraps in AsyncToolAdapter.
    
    Args:
        tool: Tool instance to check/wrap
        
    Returns:
        Async-compatible tool
        
    Example:
        tool = legacy_sync_tool_instance
        async_tool = ensure_tool_async(tool)
        await async_tool.execute(query="test")
    """
    # Check if tool already has async execute
    if hasattr(tool, 'execute') and inspect.iscoroutinefunction(tool.execute):
        return tool
    
    # Check if tool has async initialize
    if hasattr(tool, 'initialize') and inspect.iscoroutinefunction(tool.initialize):
        return tool
    
    # Wrap in adapter
    logger.info(f"Wrapping tool '{getattr(tool, 'name', 'unknown')}' with async adapter")
    return AsyncToolAdapter(tool)


def cleanup_executor() -> None:
    """
    Cleanup thread pool executor.
    Call during application shutdown.
    
    FIXED: Thread-safe cleanup with proper locking.
    """
    global _executor, _executor_shutdown
    
    with _executor_lock:
        if _executor and not _executor_shutdown:
            logger.info("Shutting down tool adapter thread pool...")
            
            try:
                # Shutdown executor gracefully
                _executor.shutdown(wait=True, cancel_futures=False)
                logger.info("✓ Tool adapter thread pool shut down successfully")
            except Exception as e:
                logger.error(f"Error shutting down thread pool: {e}")
            finally:
                _executor = None
                _executor_shutdown = True


async def async_cleanup_executor() -> None:
    """
    Async wrapper for cleanup_executor.
    Runs cleanup in thread pool to avoid blocking.
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, cleanup_executor)


# Export public API
__all__ = [
    'sync_to_async_adapter',
    'ensure_async',
    'AsyncToolAdapter',
    'ensure_tool_async',
    'cleanup_executor',
    'async_cleanup_executor',
    'get_executor'
]
```

---

## File 7: `backend/app/tools/registry.py` (Enhanced - Complete Replacement)

```python
"""
Tool Registry for dynamic tool instantiation and management.
Enables config-driven tool registration with dependency injection.

Version: 2.0.0 (Enhanced error handling and validation)
"""
import logging
from typing import Dict, Any, List, Optional, Callable, Type
from dataclasses import dataclass
import asyncio

from ..config import settings
from ..config.tool_settings import tool_settings, ToolSettings
from .base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ToolDependencies:
    """
    Container for tool dependencies.
    Provides dependency injection for tool initialization.
    """
    settings: Any
    tool_settings: ToolSettings
    db_session_maker: Optional[Any] = None
    cache_service: Optional[Any] = None
    http_client: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for factory functions."""
        return {
            'settings': self.settings,
            'tool_settings': self.tool_settings,
            'db_session_maker': self.db_session_maker,
            'cache_service': self.cache_service,
            'http_client': self.http_client
        }


class ToolFactory:
    """
    Factory for creating tool instances.
    Encapsulates tool creation logic with dependency injection.
    """
    
    @staticmethod
    def create_rag_tool(dependencies: ToolDependencies) -> BaseTool:
        """Create RAG tool instance."""
        from .rag_tool import RAGTool
        
        tool = RAGTool()
        logger.debug("RAG tool created (initialization deferred to async)")
        return tool
    
    @staticmethod
    def create_memory_tool(dependencies: ToolDependencies) -> BaseTool:
        """Create Memory tool instance."""
        from .memory_tool import MemoryTool
        
        tool = MemoryTool()
        logger.debug("Memory tool created (initialization deferred to async)")
        return tool
    
    @staticmethod
    def create_escalation_tool(dependencies: ToolDependencies) -> BaseTool:
        """Create Escalation tool instance."""
        from .escalation_tool import EscalationTool
        
        tool = EscalationTool()
        logger.debug("Escalation tool created (initialization deferred to async)")
        return tool
    
    @staticmethod
    def create_attachment_tool(dependencies: ToolDependencies) -> BaseTool:
        """Create Attachment tool instance."""
        from .attachment_tool import AttachmentTool
        
        tool = AttachmentTool()
        logger.debug("Attachment tool created (initialization deferred to async)")
        return tool
    
    @staticmethod
    def create_crm_tool(dependencies: ToolDependencies) -> BaseTool:
        """Create CRM tool instance."""
        from .crm_tool import CRMTool
        
        tool = CRMTool()
        logger.debug("CRM tool created (initialization deferred to async)")
        return tool


class ToolRegistry:
    """
    Central registry for tool management.
    Handles tool creation, initialization, and lifecycle.
    
    Version 2.0.0: Enhanced error handling and validation.
    """
    
    # Registry of tool factories
    _factories: Dict[str, Callable[[ToolDependencies], BaseTool]] = {
        'rag': ToolFactory.create_rag_tool,
        'memory': ToolFactory.create_memory_tool,
        'escalation': ToolFactory.create_escalation_tool,
        'attachment': ToolFactory.create_attachment_tool,
        'crm': ToolFactory.create_crm_tool,
    }
    
    @classmethod
    def register(
        cls,
        name: str,
        factory: Callable[[ToolDependencies], BaseTool]
    ) -> None:
        """
        Register a tool factory.
        
        Args:
            name: Tool identifier
            factory: Factory function that creates tool instance
        """
        if name in cls._factories:
            logger.warning(f"Overwriting existing tool factory: {name}")
        
        cls._factories[name] = factory
        logger.info(f"Registered tool factory: {name}")
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a tool factory.
        
        Args:
            name: Tool identifier
        """
        if name in cls._factories:
            del cls._factories[name]
            logger.info(f"Unregistered tool factory: {name}")
    
    @classmethod
    def get_factory(
        cls,
        name: str
    ) -> Optional[Callable[[ToolDependencies], BaseTool]]:
        """
        Get tool factory by name.
        
        Args:
            name: Tool identifier
            
        Returns:
            Factory function or None if not found
        """
        return cls._factories.get(name)
    
    @classmethod
    def list_available_tools(cls) -> List[str]:
        """
        List all available tool names.
        
        Returns:
            List of tool identifiers
        """
        return list(cls._factories.keys())
    
    @classmethod
    def create_tools(
        cls,
        dependencies: Optional[ToolDependencies] = None,
        enabled_only: bool = True
    ) -> Dict[str, BaseTool]:
        """
        Create tool instances based on configuration.
        
        Args:
            dependencies: Tool dependencies (uses defaults if not provided)
            enabled_only: Only create enabled tools
            
        Returns:
            Dictionary mapping tool names to instances
        """
        # Use default dependencies if not provided
        if dependencies is None:
            dependencies = ToolDependencies(
                settings=settings,
                tool_settings=tool_settings
            )
        
        tools = {}
        enabled_tools = (
            tool_settings.get_enabled_tools() if enabled_only
            else cls.list_available_tools()
        )
        
        logger.info(
            f"Creating tools (enabled_only={enabled_only}): {enabled_tools}"
        )
        
        for tool_name in enabled_tools:
            try:
                # Validate tool configuration
                warnings = tool_settings.validate_tool_config(tool_name)
                for warning in warnings:
                    logger.warning(f"Tool '{tool_name}': {warning}")
                
                # Get factory
                factory = cls.get_factory(tool_name)
                if not factory:
                    logger.error(f"No factory registered for tool: {tool_name}")
                    continue
                
                # Create tool instance
                logger.debug(f"Creating tool instance: {tool_name}")
                tool = factory(dependencies)
                
                tools[tool_name] = tool
                logger.info(f"✓ Created tool: {tool_name} ({tool.__class__.__name__})")
                
            except Exception as e:
                logger.error(
                    f"Failed to create tool '{tool_name}': {e}",
                    exc_info=True
                )
                # Continue creating other tools even if one fails
        
        logger.info(
            f"Tool creation complete: {len(tools)}/{len(enabled_tools)} tools created"
        )
        
        return tools
    
    @classmethod
    async def initialize_tools(
        cls,
        tools: Dict[str, BaseTool],
        concurrent: bool = True
    ) -> Dict[str, bool]:
        """
        Initialize all tools asynchronously.
        
        Args:
            tools: Dictionary of tool instances
            concurrent: Whether to initialize tools concurrently
            
        Returns:
            Dictionary mapping tool names to initialization success status
        """
        logger.info(
            f"Initializing {len(tools)} tools (concurrent={concurrent})..."
        )
        
        results = {}
        
        if concurrent:
            # Initialize tools concurrently
            tasks = []
            tool_names = []
            
            for tool_name, tool in tools.items():
                if hasattr(tool, 'initialize'):
                    tasks.append(tool.initialize())
                    tool_names.append(tool_name)
                else:
                    logger.warning(f"Tool '{tool_name}' has no initialize method")
                    results[tool_name] = False
            
            # Gather results
            init_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for tool_name, result in zip(tool_names, init_results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Failed to initialize tool '{tool_name}': {result}",
                        exc_info=result
                    )
                    results[tool_name] = False
                else:
                    results[tool_name] = True
        else:
            # Initialize tools sequentially
            for tool_name, tool in tools.items():
                try:
                    if hasattr(tool, 'initialize'):
                        await tool.initialize()
                        results[tool_name] = True
                        logger.info(f"✓ Initialized tool: {tool_name}")
                    else:
                        logger.warning(f"Tool '{tool_name}' has no initialize method")
                        results[tool_name] = False
                except Exception as e:
                    logger.error(
                        f"Failed to initialize tool '{tool_name}': {e}",
                        exc_info=True
                    )
                    results[tool_name] = False
        
        success_count = sum(1 for success in results.values() if success)
        logger.info(
            f"Tool initialization complete: {success_count}/{len(tools)} succeeded"
        )
        
        return results
    
    @classmethod
    async def cleanup_tools(
        cls,
        tools: Dict[str, BaseTool],
        concurrent: bool = True
    ) -> None:
        """
        Cleanup all tools asynchronously.
        
        Args:
            tools: Dictionary of tool instances
            concurrent: Whether to cleanup tools concurrently
        """
        logger.info(
            f"Cleaning up {len(tools)} tools (concurrent={concurrent})..."
        )
        
        if concurrent:
            # Cleanup tools concurrently
            tasks = []
            tool_names = []
            
            for tool_name, tool in tools.items():
                if hasattr(tool, 'cleanup'):
                    tasks.append(tool.cleanup())
                    tool_names.append(tool_name)
            
            # Gather results
            cleanup_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for tool_name, result in zip(tool_names, cleanup_results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Error cleaning up tool '{tool_name}': {result}"
                    )
        else:
            # Cleanup tools sequentially
            for tool_name, tool in tools.items():
                try:
                    if hasattr(tool, 'cleanup'):
                        await tool.cleanup()
                        logger.info(f"✓ Cleaned up tool: {tool_name}")
                except Exception as e:
                    logger.error(
                        f"Error cleaning up tool '{tool_name}': {e}",
                        exc_info=True
                    )
        
        logger.info("Tool cleanup complete")
    
    @classmethod
    async def create_and_initialize_tools(
        cls,
        dependencies: Optional[ToolDependencies] = None,
        enabled_only: bool = True,
        concurrent_init: bool = True,
        fail_on_error: bool = False
    ) -> Dict[str, BaseTool]:
        """
        Create and initialize tools in one step.
        
        Version 2.0.0: Added fail_on_error parameter for better control.
        
        Args:
            dependencies: Tool dependencies
            enabled_only: Only create enabled tools
            concurrent_init: Initialize tools concurrently
            fail_on_error: Raise exception if any tool fails to initialize
            
        Returns:
            Dictionary of initialized tool instances
            
        Raises:
            RuntimeError: If fail_on_error=True and any tool fails
        """
        # Create tools
        tools = cls.create_tools(dependencies, enabled_only)
        
        if not tools:
            logger.warning("No tools were created")
            return {}
        
        # Initialize tools
        init_results = await cls.initialize_tools(tools, concurrent_init)
        
        # Check for failures
        failed_tools = [
            name for name, success in init_results.items()
            if not success
        ]
        
        if failed_tools:
            error_msg = f"Failed to initialize tools: {', '.join(failed_tools)}"
            logger.error(error_msg)
            
            if fail_on_error:
                # Clean up successfully initialized tools
                successful_tools = {
                    name: tool for name, tool in tools.items()
                    if init_results.get(name, False)
                }
                await cls.cleanup_tools(successful_tools)
                
                raise RuntimeError(error_msg)
            else:
                # Remove failed tools but continue
                logger.warning(
                    f"Continuing with {len(tools) - len(failed_tools)}/{len(tools)} tools. "
                    f"Failed: {failed_tools}"
                )
                for tool_name in failed_tools:
                    tools.pop(tool_name, None)
        
        return tools
    
    @classmethod
    def get_tool_info(cls, tool_name: str) -> Dict[str, Any]:
        """
        Get information about a registered tool.
        
        Args:
            tool_name: Tool identifier
            
        Returns:
            Dictionary with tool information
        """
        factory = cls.get_factory(tool_name)
        
        if not factory:
            return {
                "registered": False,
                "name": tool_name
            }
        
        enabled_tools = tool_settings.get_enabled_tools()
        
        return {
            "registered": True,
            "name": tool_name,
            "enabled": tool_name in enabled_tools,
            "factory": factory.__name__,
            "config": tool_settings.get_tool_config(tool_name),
            "warnings": tool_settings.validate_tool_config(tool_name)
        }
    
    @classmethod
    def get_registry_status(cls) -> Dict[str, Any]:
        """
        Get status of the entire registry.
        
        Returns:
            Dictionary with registry status information
        """
        available_tools = cls.list_available_tools()
        enabled_tools = tool_settings.get_enabled_tools()
        
        return {
            "total_available": len(available_tools),
            "total_enabled": len(enabled_tools),
            "available_tools": available_tools,
            "enabled_tools": enabled_tools,
            "disabled_tools": [
                t for t in available_tools
                if t not in enabled_tools
            ],
            "tools": {
                tool_name: cls.get_tool_info(tool_name)
                for tool_name in available_tools
            }
        }
    
    @classmethod
    def validate_registry(cls) -> List[str]:
        """
        Validate registry configuration.
        
        Returns:
            List of validation warnings/errors
        """
        issues = []
        
        enabled_tools = tool_settings.get_enabled_tools()
        available_tools = cls.list_available_tools()
        
        # Check for enabled tools that don't have factories
        for tool_name in enabled_tools:
            if tool_name not in available_tools:
                issues.append(
                    f"Tool '{tool_name}' is enabled but has no registered factory"
                )
        
        # Check for configuration issues
        for tool_name in enabled_tools:
            warnings = tool_settings.validate_tool_config(tool_name)
            for warning in warnings:
                issues.append(f"{tool_name}: {warning}")
        
        return issues


# Export public API
__all__ = [
    'ToolRegistry',
    'ToolFactory',
    'ToolDependencies'
]
```

I'll continue with the remaining files in the next response. Should I proceed with the tool implementations (RAG, Memory, Escalation, Attachment) and the final __init__.py?
