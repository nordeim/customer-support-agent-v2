"""
Base tool class for Microsoft Agent Framework tools.
All tools should inherit from this base class.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Abstract base class for agent tools.
    Provides common interface and functionality for all tools.
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
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize tool resources."""
        try:
            self._setup()
            self.initialized = True
            logger.info(f"Tool '{self.name}' initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize tool '{self.name}': {e}")
            raise
    
    @abstractmethod
    def _setup(self) -> None:
        """Setup tool-specific resources. Override in subclasses."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute tool action.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            Dictionary with execution results
        """
        pass
    
    async def __call__(self, **kwargs) -> Dict[str, Any]:
        """
        Make tool callable for agent framework.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            Dictionary with execution results
        """
        if not self.initialized:
            raise RuntimeError(f"Tool '{self.name}' not initialized")
        
        try:
            logger.debug(f"Executing tool '{self.name}' with params: {list(kwargs.keys())}")
            result = await self.execute(**kwargs)
            logger.debug(f"Tool '{self.name}' execution completed")
            return result
        except Exception as e:
            logger.error(f"Tool '{self.name}' execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": self.name
            }
    
    async def cleanup(self) -> None:
        """Cleanup tool resources. Override if needed."""
        pass
