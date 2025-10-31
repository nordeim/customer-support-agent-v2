"""
Base tool class with async-first interface and comprehensive observability.
All tools should inherit from this base class.

Version: 3.0.0 (Enhanced with metrics, health checks, and validation)

Changes from 2.0.0:
- Added execution metrics collection and aggregation
- Added health check support for monitoring
- Added parameter validation hook with Pydantic support
- Added OpenAI function calling schema generation
- Added async context manager support
- Added structured error codes
- Enhanced ToolResult with error codes
- All changes are backward compatible

Migration from 2.0.0:
- No changes required - all enhancements are opt-in
- Existing tools work without modification
- To use new features, see examples in docstrings
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Type, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import functools
import warnings
import time
import asyncio
from collections import deque

logger = logging.getLogger(__name__)

# Constants
VERSION = "3.0.0"
DEFAULT_MAX_METRICS_HISTORY = 100


# ===========================
# Conditional Imports
# ===========================

try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    BaseModel = None
    ValidationError = None
    PYDANTIC_AVAILABLE = False


# ===========================
# Deprecation Decorator
# ===========================

def deprecated(message: str = "", version: str = ""):
    """
    Decorator to mark methods as deprecated.
    
    Args:
        message: Deprecation message
        version: Version when deprecated
        
    Example:
        @deprecated("Use async_method() instead", version="2.0.0")
        def old_method(self):
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            full_message = f"{func.__name__} is deprecated"
            if version:
                full_message += f" (since version {version})"
            if message:
                full_message += f". {message}"
            
            warnings.warn(
                full_message,
                category=DeprecationWarning,
                stacklevel=2
            )
            logger.warning(full_message)
            
            return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            full_message = f"{func.__name__} is deprecated"
            if version:
                full_message += f" (since version {version})"
            if message:
                full_message += f". {message}"
            
            warnings.warn(
                full_message,
                category=DeprecationWarning,
                stacklevel=2
            )
            logger.warning(full_message)
            
            return await func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# ===========================
# Enums
# ===========================

class ToolStatus(str, Enum):
    """Tool execution status."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


class ErrorCode(str, Enum):
    """
    Standardized error codes for tool operations.
    
    Enables better error handling and categorization.
    
    Example:
        result = ToolResult.error_result(
            error="Invalid parameter",
            error_code=ErrorCode.VALIDATION_ERROR
        )
    """
    # Initialization errors
    INITIALIZATION_ERROR = "initialization_error"
    CONFIGURATION_ERROR = "configuration_error"
    DEPENDENCY_ERROR = "dependency_error"
    
    # Validation errors
    VALIDATION_ERROR = "validation_error"
    PARAMETER_ERROR = "parameter_error"
    SCHEMA_ERROR = "schema_error"
    
    # Execution errors
    EXECUTION_ERROR = "execution_error"
    TIMEOUT_ERROR = "timeout_error"
    RESOURCE_ERROR = "resource_error"
    
    # External service errors
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    
    # Data errors
    NOT_FOUND_ERROR = "not_found_error"
    ALREADY_EXISTS_ERROR = "already_exists_error"
    DATA_ERROR = "data_error"
    
    # System errors
    CIRCUIT_BREAKER_ERROR = "circuit_breaker_error"
    UNKNOWN_ERROR = "unknown_error"
    
    @property
    def is_retryable(self) -> bool:
        """
        Determine if this error type is retryable.
        
        Returns:
            True if the operation should be retried
        """
        retryable = {
            self.TIMEOUT_ERROR,
            self.NETWORK_ERROR,
            self.RATE_LIMIT_ERROR,
            self.RESOURCE_ERROR,
            self.API_ERROR  # Some API errors are retryable
        }
        return self in retryable
    
    @property
    def is_client_error(self) -> bool:
        """
        Determine if this is a client error (4xx equivalent).
        
        Returns:
            True if error is due to client input
        """
        client_errors = {
            self.VALIDATION_ERROR,
            self.PARAMETER_ERROR,
            self.SCHEMA_ERROR,
            self.AUTHENTICATION_ERROR,
            self.AUTHORIZATION_ERROR,
            self.NOT_FOUND_ERROR
        }
        return self in client_errors


# ===========================
# ToolResult Data Structure
# ===========================

@dataclass
class ToolResult:
    """
    Standardized return type for all tool operations.
    
    Version 3.0.0: Enhanced with error_code support.
    
    Attributes:
        success: Whether the operation succeeded
        data: Operation result data (tool-specific structure)
        metadata: Additional context (timestamps, tool version, etc.)
        error: Error message if success=False
        status: Detailed status (SUCCESS, ERROR, PARTIAL)
        error_code: Structured error code (NEW in 3.0.0)
    
    Example:
        # Success result
        result = ToolResult.success_result(
            data={"results": [...]},
            metadata={"tool": "rag", "count": 5}
        )
        
        # Error result with code
        result = ToolResult.error_result(
            error="Invalid query parameter",
            error_code=ErrorCode.VALIDATION_ERROR,
            metadata={"param": "query"}
        )
    """
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    status: ToolStatus = ToolStatus.SUCCESS
    error_code: Optional[ErrorCode] = None  # NEW in 3.0.0
    
    def __post_init__(self):
        """Validate and normalize status."""
        if not self.success and self.status == ToolStatus.SUCCESS:
            self.status = ToolStatus.ERROR
        
        if self.error and not self.metadata.get('error_type'):
            if isinstance(self.error, Exception):
                self.metadata['error_type'] = type(self.error).__name__
            else:
                self.metadata['error_type'] = 'unknown'
        
        # Add error_code to metadata for backward compatibility
        if self.error_code and 'error_code' not in self.metadata:
            self.metadata['error_code'] = self.error_code.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "success": self.success,
            "data": self.data,
            "metadata": self.metadata,
            "error": self.error,
            "status": self.status.value
        }
        
        # Include error_code if present
        if self.error_code:
            result["error_code"] = self.error_code.value
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolResult':
        """Create ToolResult from dictionary."""
        error_code = None
        if 'error_code' in data:
            try:
                error_code = ErrorCode(data['error_code'])
            except ValueError:
                pass
        
        return cls(
            success=data.get('success', False),
            data=data.get('data', {}),
            metadata=data.get('metadata', {}),
            error=data.get('error'),
            status=ToolStatus(data.get('status', 'error')),
            error_code=error_code
        )
    
    @classmethod
    def success_result(
        cls,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'ToolResult':
        """Create a successful result."""
        return cls(
            success=True,
            data=data,
            metadata=metadata or {},
            status=ToolStatus.SUCCESS
        )
    
    @classmethod
    def error_result(
        cls,
        error: str,
        metadata: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        error_code: Optional[ErrorCode] = None
    ) -> 'ToolResult':
        """
        Create an error result.
        
        Args:
            error: Error message
            metadata: Additional context
            data: Partial data if available
            error_code: Structured error code (NEW in 3.0.0)
        """
        return cls(
            success=False,
            error=error,
            data=data or {},
            metadata=metadata or {},
            status=ToolStatus.ERROR,
            error_code=error_code
        )
    
    @classmethod
    def partial_result(
        cls,
        data: Dict[str, Any],
        error: str,
        metadata: Optional[Dict[str, Any]] = None,
        error_code: Optional[ErrorCode] = None
    ) -> 'ToolResult':
        """Create a partial success result."""
        return cls(
            success=False,
            data=data,
            error=error,
            metadata=metadata or {},
            status=ToolStatus.PARTIAL,
            error_code=error_code
        )


# ===========================
# ToolMetrics Data Structure
# ===========================

@dataclass
class ToolMetrics:
    """
    Execution metrics for tool performance monitoring.
    
    NEW in 3.0.0.
    
    Attributes:
        tool_name: Name of the tool
        operation: Operation performed
        success: Whether execution succeeded
        duration_ms: Execution duration in milliseconds
        timestamp: Unix timestamp of execution
        error_type: Type of error if failed
        error_code: Structured error code if failed
    
    Example:
        metrics = ToolMetrics(
            tool_name="rag_search",
            operation="search",
            success=True,
            duration_ms=45.2,
            timestamp=time.time()
        )
    """
    tool_name: str
    operation: str
    success: bool
    duration_ms: float
    timestamp: float
    error_type: Optional[str] = None
    error_code: Optional[ErrorCode] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "tool_name": self.tool_name,
            "operation": self.operation,
            "success": self.success,
            "duration_ms": round(self.duration_ms, 2),
            "timestamp": self.timestamp
        }
        
        if self.error_type:
            result["error_type"] = self.error_type
        
        if self.error_code:
            result["error_code"] = self.error_code.value
        
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        status = "✓" if self.success else "✗"
        return (
            f"<ToolMetrics {status} {self.tool_name}.{self.operation} "
            f"{self.duration_ms:.1f}ms>"
        )


# ===========================
# BaseTool (Async-First with Enhanced Observability)
# ===========================

class BaseTool(ABC):
    """
    Abstract base class for agent tools with async-first interface.
    
    Version 3.0.0: Enhanced with metrics, health checks, and validation.
    
    New Features (all backward compatible):
    - Execution metrics collection and aggregation
    - Health check support for monitoring
    - Parameter validation with Pydantic support
    - OpenAI function calling schema generation
    - Async context manager support
    - Structured error codes
    
    Subclasses must implement:
    - async initialize(): Setup resources (async-safe)
    - async cleanup(): Cleanup resources
    - async execute(**kwargs) -> ToolResult: Main execution logic
    
    Optional overrides:
    - async _custom_health_check(): Custom health checks
    - _get_parameters_schema(): OpenAI function parameters
    - _get_required_parameters(): Required parameter names
    
    Example:
        class MyTool(BaseTool):
            def __init__(self):
                super().__init__(
                    name="my_tool",
                    description="Does something useful",
                    version="1.0.0"
                )
            
            async def initialize(self):
                # Setup resources
                pass
            
            async def cleanup(self):
                # Cleanup resources
                pass
            
            async def execute(self, **kwargs) -> ToolResult:
                # Implementation
                return ToolResult.success_result(data={...})
    
    Using async context manager (NEW in 3.0.0):
        async with MyTool() as tool:
            result = await tool(param="value")
        # Auto-cleanup on exit
    """
    
    # Subclasses can set this to enable automatic validation
    request_model: Optional[Type] = None
    
    def __init__(self, name: str, description: str, version: str = "1.0.0"):
        """
        Initialize base tool.
        
        Args:
            name: Unique tool identifier
            description: Human-readable tool description
            version: Tool version
        """
        self.name = name
        self.description = description
        self.version = version
        self.initialized = False
        
        # NEW in 3.0.0: Metrics collection
        self._metrics_enabled = True
        self._metrics_history: deque = deque(maxlen=DEFAULT_MAX_METRICS_HISTORY)
        self._metrics_lock = asyncio.Lock()
        
        logger.debug(f"Tool '{name}' created (version {version})")
    
    # ===========================
    # Abstract Methods (Must Implement)
    # ===========================
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize tool resources (async-safe).
        
        Called during tool registration or agent startup.
        Should set up:
        - Database connections
        - HTTP clients
        - Model loading
        - Cache connections
        
        Raises:
            Exception: If initialization fails
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup tool resources.
        
        Called during agent shutdown.
        Should cleanup:
        - Close connections
        - Release memory
        - Flush caches
        """
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute tool action (async-first).
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult with execution outcome
            
        Raises:
            Exception: Tool-specific errors (wrapped in ToolResult if possible)
        """
        pass
    
    # ===========================
    # Main Execution Method
    # ===========================
    
    async def __call__(self, **kwargs) -> ToolResult:
        """
        Make tool callable (always returns ToolResult).
        
        Version 3.0.0: Enhanced with validation and metrics.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult with execution outcome
        """
        start_time = time.time()
        operation = kwargs.get("action", "execute")
        result = None
        
        # Check initialization
        if not self.initialized:
            result = ToolResult.error_result(
                error=f"Tool '{self.name}' not initialized. Call await tool.initialize() first.",
                metadata={"tool": self.name, "initialized": False},
                error_code=ErrorCode.INITIALIZATION_ERROR
            )
            
            # Record metrics even for initialization error
            if self._metrics_enabled:
                await self._record_metrics(
                    operation=operation,
                    success=False,
                    duration_ms=(time.time() - start_time) * 1000,
                    error_type="InitializationError",
                    error_code=ErrorCode.INITIALIZATION_ERROR
                )
            
            return result
        
        try:
            logger.debug(f"Executing tool '{self.name}' with params: {list(kwargs.keys())}")
            
            # NEW in 3.0.0: Parameter validation
            validation_result = await self.validate_params(**kwargs)
            if not validation_result.success:
                # Record validation failure metrics
                if self._metrics_enabled:
                    await self._record_metrics(
                        operation=operation,
                        success=False,
                        duration_ms=(time.time() - start_time) * 1000,
                        error_type="ValidationError",
                        error_code=ErrorCode.VALIDATION_ERROR
                    )
                
                return validation_result
            
            # Call execute() which should return ToolResult
            result = await self.execute(**kwargs)
            
            # Ensure result is always ToolResult
            if not isinstance(result, ToolResult):
                # Wrap legacy dict format
                if isinstance(result, dict):
                    if "success" in result:
                        result = ToolResult(
                            success=result.get("success", False),
                            data=result.get("data", result),
                            error=result.get("error"),
                            metadata=result.get("metadata", {"tool": self.name})
                        )
                    else:
                        result = ToolResult.success_result(
                            data=result,
                            metadata={"tool": self.name}
                        )
                else:
                    result = ToolResult.success_result(
                        data={"result": result},
                        metadata={"tool": self.name}
                    )
            
            logger.debug(f"Tool '{self.name}' execution completed: {result.status.value}")
            
            # NEW in 3.0.0: Record metrics
            if self._metrics_enabled:
                await self._record_metrics(
                    operation=operation,
                    success=result.success,
                    duration_ms=(time.time() - start_time) * 1000,
                    error_type=result.metadata.get('error_type'),
                    error_code=result.error_code
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Tool '{self.name}' execution failed: {e}", exc_info=True)
            
            result = ToolResult.error_result(
                error=str(e),
                metadata={
                    "tool": self.name,
                    "error_type": type(e).__name__
                },
                error_code=ErrorCode.EXECUTION_ERROR
            )
            
            # Record error metrics
            if self._metrics_enabled:
                await self._record_metrics(
                    operation=operation,
                    success=False,
                    duration_ms=(time.time() - start_time) * 1000,
                    error_type=type(e).__name__,
                    error_code=ErrorCode.EXECUTION_ERROR
                )
            
            return result
    
    # ===========================
    # Validation (NEW in 3.0.0)
    # ===========================
    
    async def validate_params(self, **kwargs) -> ToolResult:
        """
        Validate parameters before execution.
        
        NEW in 3.0.0.
        
        If request_model is set (Pydantic BaseModel), validates kwargs against it.
        Subclasses can override for custom validation logic.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            ToolResult.success_result() if valid, error_result() otherwise
        
        Example:
            class MyTool(BaseTool):
                request_model = MyRequestModel  # Pydantic model
                
                # Validation happens automatically
        
        Example (custom validation):
            class MyTool(BaseTool):
                async def validate_params(self, **kwargs):
                    if 'required_param' not in kwargs:
                        return ToolResult.error_result(
                            error="required_param is missing",
                            error_code=ErrorCode.PARAMETER_ERROR
                        )
                    return ToolResult.success_result(data={})
        """
        # Check if Pydantic model is set
        if self.request_model and PYDANTIC_AVAILABLE:
            try:
                # Validate against Pydantic model
                self.request_model(**kwargs)
            except ValidationError as e:
                return ToolResult.error_result(
                    error=f"Parameter validation failed: {e}",
                    metadata={
                        "tool": self.name,
                        "validation_errors": e.errors()
                    },
                    error_code=ErrorCode.VALIDATION_ERROR
                )
            except Exception as e:
                return ToolResult.error_result(
                    error=f"Validation error: {str(e)}",
                    metadata={"tool": self.name},
                    error_code=ErrorCode.VALIDATION_ERROR
                )
        
        # No validation needed or Pydantic not available
        return ToolResult.success_result(data={})
    
    # ===========================
    # Health Check (NEW in 3.0.0)
    # ===========================
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check tool health status.
        
        NEW in 3.0.0.
        
        Returns health information for monitoring and debugging.
        Subclasses can override _custom_health_check() for custom checks.
        
        Returns:
            Health status dictionary
        
        Example response:
            {
                "tool": "rag_search",
                "version": "3.0.0",
                "initialized": True,
                "healthy": True,
                "latency_ms": 2.3,
                "metrics": {...},
                "custom": {...}  # From _custom_health_check()
            }
        
        Example (custom health check):
            class MyTool(BaseTool):
                async def _custom_health_check(self):
                    # Check database connection
                    db_ok = await self.db.ping()
                    return {
                        "database": "healthy" if db_ok else "unhealthy",
                        "connections": self.db.pool_size
                    }
        """
        start_time = time.time()
        
        health = {
            "tool": self.name,
            "version": self.version,
            "initialized": self.initialized,
            "healthy": self.initialized
        }
        
        # Get metrics summary if available
        if self._metrics_enabled and self._metrics_history:
            metrics_summary = self.get_metrics()
            health["metrics"] = {
                "executions": metrics_summary.get("executions", 0),
                "success_rate": metrics_summary.get("success_rate", 0.0),
                "avg_duration_ms": metrics_summary.get("avg_duration_ms", 0.0)
            }
        
        # Call custom health check
        try:
            custom_health = await self._custom_health_check()
            if custom_health:
                health["custom"] = custom_health
        except Exception as e:
            health["healthy"] = False
            health["custom_check_error"] = str(e)
            logger.error(f"Custom health check failed for '{self.name}': {e}")
        
        # Add health check latency
        health["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        
        return health
    
    async def _custom_health_check(self) -> Optional[Dict[str, Any]]:
        """
        Override this method to add custom health checks.
        
        NEW in 3.0.0.
        
        Returns:
            Custom health information or None
        
        Example:
            async def _custom_health_check(self):
                # Check external dependencies
                api_ok = await self.api_client.ping()
                cache_ok = await self.cache.ping()
                
                return {
                    "api": "healthy" if api_ok else "unhealthy",
                    "cache": "healthy" if cache_ok else "unhealthy",
                    "cache_size": await self.cache.size()
                }
        """
        return None
    
    # ===========================
    # Metrics Collection (NEW in 3.0.0)
    # ===========================
    
    async def _record_metrics(
        self,
        operation: str,
        success: bool,
        duration_ms: float,
        error_type: Optional[str] = None,
        error_code: Optional[ErrorCode] = None
    ) -> None:
        """
        Record execution metrics (internal method).
        
        NEW in 3.0.0.
        
        Thread-safe metrics recording with bounded memory.
        """
        if not self._metrics_enabled:
            return
        
        try:
            metrics = ToolMetrics(
                tool_name=self.name,
                operation=operation,
                success=success,
                duration_ms=duration_ms,
                timestamp=time.time(),
                error_type=error_type,
                error_code=error_code
            )
            
            # Thread-safe append (deque is thread-safe for append/pop)
            # But use lock for consistency
            async with self._metrics_lock:
                self._metrics_history.append(metrics)
            
            logger.debug(f"Recorded metrics: {metrics}")
            
        except Exception as e:
            # Don't fail execution due to metrics errors
            logger.error(f"Failed to record metrics: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated execution metrics.
        
        NEW in 3.0.0.
        
        Returns:
            Aggregated metrics dictionary
        
        Example response:
            {
                "executions": 150,
                "success_rate": 0.96,
                "avg_duration_ms": 45.3,
                "min_duration_ms": 12.1,
                "max_duration_ms": 234.5,
                "p50_duration_ms": 42.0,
                "p95_duration_ms": 89.3,
                "last_execution": 1704123456.789,
                "errors_by_type": {"ValidationError": 3, "TimeoutError": 3},
                "errors_by_code": {"validation_error": 3, "timeout_error": 3}
            }
        """
        if not self._metrics_history:
            return {"executions": 0}
        
        # Convert deque to list for analysis
        metrics_list = list(self._metrics_history)
        
        total = len(metrics_list)
        successes = sum(1 for m in metrics_list if m.success)
        durations = [m.duration_ms for m in metrics_list]
        
        # Calculate percentiles
        sorted_durations = sorted(durations)
        p50_idx = int(len(sorted_durations) * 0.50)
        p95_idx = int(len(sorted_durations) * 0.95)
        
        # Error analysis
        errors_by_type = {}
        errors_by_code = {}
        
        for m in metrics_list:
            if not m.success:
                if m.error_type:
                    errors_by_type[m.error_type] = errors_by_type.get(m.error_type, 0) + 1
                if m.error_code:
                    errors_by_code[m.error_code.value] = errors_by_code.get(m.error_code.value, 0) + 1
        
        return {
            "tool": self.name,
            "executions": total,
            "success_rate": round(successes / total, 4) if total > 0 else 0.0,
            "avg_duration_ms": round(sum(durations) / len(durations), 2) if durations else 0.0,
            "min_duration_ms": round(min(durations), 2) if durations else 0.0,
            "max_duration_ms": round(max(durations), 2) if durations else 0.0,
            "p50_duration_ms": round(sorted_durations[p50_idx], 2) if sorted_durations else 0.0,
            "p95_duration_ms": round(sorted_durations[p95_idx], 2) if sorted_durations else 0.0,
            "last_execution": metrics_list[-1].timestamp if metrics_list else None,
            "errors_by_type": errors_by_type,
            "errors_by_code": errors_by_code
        }
    
    def reset_metrics(self) -> None:
        """
        Reset metrics history.
        
        NEW in 3.0.0.
        
        Useful for testing or starting fresh measurements.
        """
        self._metrics_history.clear()
        logger.info(f"Metrics reset for tool '{self.name}'")
    
    def enable_metrics(self, enabled: bool = True) -> None:
        """
        Enable or disable metrics collection.
        
        NEW in 3.0.0.
        
        Args:
            enabled: True to enable, False to disable
        """
        self._metrics_enabled = enabled
        logger.info(f"Metrics {'enabled' if enabled else 'disabled'} for tool '{self.name}'")
    
    # ===========================
    # OpenAI Function Schema (NEW in 3.0.0)
    # ===========================
    
    def get_openai_schema(self) -> Dict[str, Any]:
        """
        Get OpenAI function calling schema for this tool.
        
        NEW in 3.0.0.
        
        Generates schema compatible with OpenAI's function calling API.
        Override _get_parameters_schema() and _get_required_parameters()
        for custom schemas.
        
        Returns:
            OpenAI function schema
        
        Example response:
            {
                "name": "rag_search",
                "description": "Search knowledge base...",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results"
                        }
                    },
                    "required": ["query"]
                }
            }
        
        Example (override in subclass):
            def _get_parameters_schema(self):
                return {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results",
                        "default": 5
                    }
                }
            
            def _get_required_parameters(self):
                return ["query"]
        """
        # Auto-generate from Pydantic model if available
        if self.request_model and PYDANTIC_AVAILABLE:
            try:
                # Get Pydantic schema
                pydantic_schema = self.request_model.model_json_schema()
                
                # Convert to OpenAI format
                return {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {
                        "type": "object",
                        "properties": pydantic_schema.get("properties", {}),
                        "required": pydantic_schema.get("required", [])
                    }
                }
            except Exception as e:
                logger.warning(f"Failed to auto-generate schema from Pydantic model: {e}")
        
        # Manual schema from overrides
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self._get_parameters_schema(),
                "required": self._get_required_parameters()
            }
        }
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """
        Get parameters schema (override in subclasses).
        
        NEW in 3.0.0.
        
        Returns:
            Dictionary mapping parameter names to JSON Schema
        
        Example:
            def _get_parameters_schema(self):
                return {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                        "minLength": 1,
                        "maxLength": 500
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10
                    }
                }
        """
        return {}
    
    def _get_required_parameters(self) -> List[str]:
        """
        Get list of required parameter names (override in subclasses).
        
        NEW in 3.0.0.
        
        Returns:
            List of required parameter names
        
        Example:
            def _get_required_parameters(self):
                return ["query", "session_id"]
        """
        return []
    
    # ===========================
    # Async Context Manager (NEW in 3.0.0)
    # ===========================
    
    async def __aenter__(self):
        """
        Enter async context manager.
        
        NEW in 3.0.0.
        
        Automatically initializes tool if not already initialized.
        
        Example:
            async with MyTool() as tool:
                result = await tool(param="value")
            # Cleanup happens automatically
        """
        if not self.initialized:
            logger.debug(f"Auto-initializing tool '{self.name}' via context manager")
            await self.initialize()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit async context manager.
        
        NEW in 3.0.0.
        
        Automatically cleans up tool resources.
        """
        try:
            logger.debug(f"Auto-cleaning up tool '{self.name}' via context manager")
            await self.cleanup()
        except Exception as e:
            logger.error(f"Error during context manager cleanup for '{self.name}': {e}")
        
        # Don't suppress exceptions
        return False
    
    # ===========================
    # Legacy Support (Deprecated)
    # ===========================
    
    @deprecated("Override async initialize() instead", version="2.0.0")
    def _setup(self) -> None:
        """
        DEPRECATED: Legacy sync setup.
        Override async initialize() instead.
        """
        pass
    
    @deprecated("Override async cleanup() instead", version="2.0.0")
    def _cleanup(self) -> None:
        """
        DEPRECATED: Legacy sync cleanup.
        Override async cleanup() instead.
        """
        pass
    
    # ===========================
    # Helper Methods
    # ===========================
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get tool information.
        
        Version 3.0.0: Enhanced with metrics and health summary.
        
        Returns:
            Dictionary with tool metadata
        """
        info = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "initialized": self.initialized,
            "metrics_enabled": self._metrics_enabled
        }
        
        # Add metrics summary if available
        if self._metrics_enabled and self._metrics_history:
            metrics = self.get_metrics()
            info["metrics_summary"] = {
                "executions": metrics.get("executions", 0),
                "success_rate": metrics.get("success_rate", 0.0),
                "avg_duration_ms": metrics.get("avg_duration_ms", 0.0)
            }
        
        return info
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<{self.__class__.__name__}(name='{self.name}', version='{self.version}')>"


# Export public API
__all__ = [
    'BaseTool',
    'ToolResult',
    'ToolStatus',
    'ToolMetrics',
    'ErrorCode',
    'deprecated'
]
