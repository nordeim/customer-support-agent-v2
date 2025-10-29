"""
Base tool class with async-first interface and ToolResult return type.
All tools should inherit from this base class.

Phase 1 Update: Introduces async contract while maintaining backward compatibility.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ===========================
# ToolResult Data Structure
# ===========================

class ToolStatus(str, Enum):
    """Tool execution status."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"  # Partial success with warnings


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
            self.metadata['error_type'] = type(self.error).__name__ if isinstance(self.error, Exception) else 'unknown'
    
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
    def success_result(cls, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> 'ToolResult':
        """Create a successful result."""
        return cls(
            success=True,
            data=data,
            metadata=metadata or {},
            status=ToolStatus.SUCCESS
        )
    
    @classmethod
    def error_result(cls, error: str, metadata: Optional[Dict[str, Any]] = None) -> 'ToolResult':
        """Create an error result."""
        return cls(
            success=False,
            error=error,
            metadata=metadata or {},
            status=ToolStatus.ERROR
        )


# ===========================
# BaseTool (Async-First)
# ===========================

class BaseTool(ABC):
    """
    Abstract base class for agent tools with async-first interface.
    
    Phase 1 Contract:
    - All tool initialization and cleanup is async
    - Primary execution method returns ToolResult
    - Legacy sync methods marked deprecated but functional
    
    Subclasses must implement:
    - async initialize(): Setup resources (async-safe)
    - async cleanup(): Cleanup resources
    - async execute(**kwargs) -> ToolResult: Main execution logic
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize base tool.
        
        Args:
            name: Unique tool identifier
            description: Human-readable tool description
        """
        self.name = name
        self.description = description
        self.initialized = False
        
        # Legacy support: call old _initialize if subclass hasn't migrated
        if hasattr(self, '_setup') and not hasattr(self, 'initialize'):
            logger.warning(
                f"Tool '{name}' uses deprecated _setup method. "
                f"Migrate to async initialize() for Phase 2+"
            )
            self._initialize()  # Legacy sync initialization
    
    def _initialize(self) -> None:
        """
        DEPRECATED: Legacy sync initialization.
        Use async initialize() instead.
        
        This method is kept for backward compatibility but will be removed in Phase 3.
        """
        try:
            if hasattr(self, '_setup'):
                self._setup()
            self.initialized = True
            logger.info(f"Tool '{self.name}' initialized (legacy mode)")
        except Exception as e:
            logger.error(f"Failed to initialize tool '{self.name}': {e}")
            raise
    
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
    
    # Legacy support methods (deprecated)
    
    def _setup(self) -> None:
        """
        DEPRECATED: Override async initialize() instead.
        Kept for backward compatibility only.
        """
        pass  # No-op, subclasses may override
    
    async def __call__(self, **kwargs) -> Dict[str, Any]:
        """
        Make tool callable for backward compatibility.
        
        DEPRECATED: Use execute() directly in new code.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            Dictionary with execution results (legacy format)
        """
        if not self.initialized:
            # Auto-initialize if using legacy sync init
            if hasattr(self, '_setup'):
                logger.warning(f"Auto-initializing '{self.name}' in legacy mode")
                self._initialize()
            else:
                raise RuntimeError(f"Tool '{self.name}' not initialized. Call await tool.initialize() first.")
        
        try:
            logger.debug(f"Executing tool '{self.name}' with params: {list(kwargs.keys())}")
            
            # Call new async execute() and convert ToolResult to legacy dict
            result = await self.execute(**kwargs)
            
            # Convert ToolResult to legacy format if needed
            if isinstance(result, ToolResult):
                legacy_result = result.to_dict()
                logger.debug(f"Tool '{self.name}' execution completed: {result.status.value}")
                return legacy_result
            else:
                # Already in dict format (legacy tool)
                logger.debug(f"Tool '{self.name}' execution completed (legacy format)")
                return result
                
        except Exception as e:
            logger.error(f"Tool '{self.name}' execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "tool": self.name
            }


# Export public API
__all__ = ['BaseTool', 'ToolResult', 'ToolStatus']
