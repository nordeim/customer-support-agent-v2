"""
Base tool class with async-first interface and ToolResult return type.
All tools should inherit from this base class.

Version: 2.0.0 (Enhanced with deprecation support and improved error handling)
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
import functools
import warnings

logger = logging.getLogger(__name__)


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
# ToolResult Data Structure
# ===========================

class ToolStatus(str, Enum):
    """Tool execution status."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


@dataclass
class ToolResult:
    """
    Standardized return type for all tool operations.
    
    Attributes:
        success: Whether the operation succeeded
        data: Operation result data (tool-specific structure)
        metadata: Additional context (timestamps, tool version, etc.)
        error: Error message if success=False
        status: Detailed status (SUCCESS, ERROR, PARTIAL)
    """
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    status: ToolStatus = ToolStatus.SUCCESS
    
    def __post_init__(self):
        """Validate and normalize status."""
        if not self.success and self.status == ToolStatus.SUCCESS:
            self.status = ToolStatus.ERROR
        
        if self.error and not self.metadata.get('error_type'):
            if isinstance(self.error, Exception):
                self.metadata['error_type'] = type(self.error).__name__
            else:
                self.metadata['error_type'] = 'unknown'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "data": self.data,
            "metadata": self.metadata,
            "error": self.error,
            "status": self.status.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolResult':
        """Create ToolResult from dictionary."""
        return cls(
            success=data.get('success', False),
            data=data.get('data', {}),
            metadata=data.get('metadata', {}),
            error=data.get('error'),
            status=ToolStatus(data.get('status', 'error'))
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
        data: Optional[Dict[str, Any]] = None
    ) -> 'ToolResult':
        """Create an error result."""
        return cls(
            success=False,
            error=error,
            data=data or {},
            metadata=metadata or {},
            status=ToolStatus.ERROR
        )
    
    @classmethod
    def partial_result(
        cls,
        data: Dict[str, Any],
        error: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'ToolResult':
        """Create a partial success result."""
        return cls(
            success=False,
            data=data,
            error=error,
            metadata=metadata or {},
            status=ToolStatus.PARTIAL
        )


# ===========================
# BaseTool (Async-First)
# ===========================

class BaseTool(ABC):
    """
    Abstract base class for agent tools with async-first interface.
    
    Version 2.0.0:
    - Enhanced with deprecation decorator
    - Improved __call__ to always return ToolResult
    - Better error handling and type safety
    - Comprehensive docstrings
    
    Subclasses must implement:
    - async initialize(): Setup resources (async-safe)
    - async cleanup(): Cleanup resources
    - async execute(**kwargs) -> ToolResult: Main execution logic
    """
    
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
        
        logger.debug(f"Tool '{name}' created (version {version})")
    
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
    
    async def __call__(self, **kwargs) -> ToolResult:
        """
        Make tool callable (always returns ToolResult).
        
        Version 2.0.0: FIXED to always return ToolResult for consistency.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult with execution outcome
        """
        if not self.initialized:
            return ToolResult.error_result(
                error=f"Tool '{self.name}' not initialized. Call await tool.initialize() first.",
                metadata={"tool": self.name, "initialized": False}
            )
        
        try:
            logger.debug(f"Executing tool '{self.name}' with params: {list(kwargs.keys())}")
            
            # Call execute() which should return ToolResult
            result = await self.execute(**kwargs)
            
            # FIXED: Ensure result is always ToolResult
            if not isinstance(result, ToolResult):
                # Wrap legacy dict format
                if isinstance(result, dict):
                    if "success" in result:
                        # Has success flag, convert
                        result = ToolResult(
                            success=result.get("success", False),
                            data=result.get("data", result),
                            error=result.get("error"),
                            metadata=result.get("metadata", {"tool": self.name})
                        )
                    else:
                        # Assume success if no error structure
                        result = ToolResult.success_result(
                            data=result,
                            metadata={"tool": self.name}
                        )
                else:
                    # Unknown type, wrap in data
                    result = ToolResult.success_result(
                        data={"result": result},
                        metadata={"tool": self.name}
                    )
            
            logger.debug(f"Tool '{self.name}' execution completed: {result.status.value}")
            return result
            
        except Exception as e:
            logger.error(f"Tool '{self.name}' execution failed: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={
                    "tool": self.name,
                    "error_type": type(e).__name__
                }
            )
    
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
        
        Returns:
            Dictionary with tool metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "initialized": self.initialized
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<{self.__class__.__name__}(name='{self.name}', version='{self.version}')>"


# Export public API
__all__ = [
    'BaseTool',
    'ToolResult',
    'ToolStatus',
    'deprecated'
]
