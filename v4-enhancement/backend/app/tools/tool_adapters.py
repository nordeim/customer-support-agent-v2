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
