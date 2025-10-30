"""
Tool Registry for dynamic tool instantiation and management.
Enables config-driven tool registration with dependency injection.

Phase 2: Centralized tool factory with feature flags and lifecycle management.
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
        """
        Create CRM tool instance (Phase 5).
        Placeholder factory for future implementation.
        """
        logger.info("CRM tool factory called but tool not yet implemented (Phase 5)")
        raise NotImplementedError(
            "CRM tool is not yet implemented. "
            "Set ENABLE_CRM_TOOL=false or implement CRMLookupTool in Phase 5."
        )
    
    @staticmethod
    def create_billing_tool(dependencies: ToolDependencies) -> BaseTool:
        """
        Create Billing tool instance (Phase 5).
        Placeholder factory for future implementation.
        """
        logger.info("Billing tool factory called but tool not yet implemented (Phase 5)")
        raise NotImplementedError(
            "Billing tool is not yet implemented. "
            "Set ENABLE_BILLING_TOOL=false or implement BillingTool in Phase 5."
        )
    
    @staticmethod
    def create_inventory_tool(dependencies: ToolDependencies) -> BaseTool:
        """
        Create Inventory tool instance (Phase 5).
        Placeholder factory for future implementation.
        """
        logger.info("Inventory tool factory called but tool not yet implemented (Phase 5)")
        raise NotImplementedError(
            "Inventory tool is not yet implemented. "
            "Set ENABLE_INVENTORY_TOOL=false or implement InventoryTool in Phase 5."
        )


class ToolRegistry:
    """
    Central registry for tool management.
    Handles tool creation, initialization, and lifecycle.
    """
    
    # Registry of tool factories
    _factories: Dict[str, Callable[[ToolDependencies], BaseTool]] = {
        'rag': ToolFactory.create_rag_tool,
        'memory': ToolFactory.create_memory_tool,
        'escalation': ToolFactory.create_escalation_tool,
        'attachment': ToolFactory.create_attachment_tool,
        'crm': ToolFactory.create_crm_tool,
        'billing': ToolFactory.create_billing_tool,
        'inventory': ToolFactory.create_inventory_tool,
    }
    
    @classmethod
    def register(cls, name: str, factory: Callable[[ToolDependencies], BaseTool]) -> None:
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
    def get_factory(cls, name: str) -> Optional[Callable[[ToolDependencies], BaseTool]]:
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
        enabled_tools = tool_settings.get_enabled_tools() if enabled_only else cls.list_available_tools()
        
        logger.info(f"Creating tools (enabled_only={enabled_only}): {enabled_tools}")
        
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
                
            except NotImplementedError as e:
                logger.warning(f"Tool '{tool_name}' not yet implemented: {e}")
            except Exception as e:
                logger.error(f"Failed to create tool '{tool_name}': {e}", exc_info=True)
                # Continue creating other tools even if one fails
        
        logger.info(f"Tool creation complete: {len(tools)}/{len(enabled_tools)} tools created")
        
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
        logger.info(f"Initializing {len(tools)} tools (concurrent={concurrent})...")
        
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
                    logger.error(f"Failed to initialize tool '{tool_name}': {result}")
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
                    logger.error(f"Failed to initialize tool '{tool_name}': {e}", exc_info=True)
                    results[tool_name] = False
        
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Tool initialization complete: {success_count}/{len(tools)} succeeded")
        
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
        logger.info(f"Cleaning up {len(tools)} tools (concurrent={concurrent})...")
        
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
                    logger.error(f"Error cleaning up tool '{tool_name}': {result}")
        else:
            # Cleanup tools sequentially
            for tool_name, tool in tools.items():
                try:
                    if hasattr(tool, 'cleanup'):
                        await tool.cleanup()
                        logger.info(f"✓ Cleaned up tool: {tool_name}")
                except Exception as e:
                    logger.error(f"Error cleaning up tool '{tool_name}': {e}", exc_info=True)
        
        logger.info("Tool cleanup complete")
    
    @classmethod
    async def create_and_initialize_tools(
        cls,
        dependencies: Optional[ToolDependencies] = None,
        enabled_only: bool = True,
        concurrent_init: bool = True
    ) -> Dict[str, BaseTool]:
        """
        Create and initialize tools in one step.
        
        Args:
            dependencies: Tool dependencies
            enabled_only: Only create enabled tools
            concurrent_init: Initialize tools concurrently
            
        Returns:
            Dictionary of initialized tool instances
        """
        # Create tools
        tools = cls.create_tools(dependencies, enabled_only)
        
        if not tools:
            logger.warning("No tools were created")
            return {}
        
        # Initialize tools
        init_results = await cls.initialize_tools(tools, concurrent_init)
        
        # Remove tools that failed to initialize
        failed_tools = [name for name, success in init_results.items() if not success]
        for tool_name in failed_tools:
            logger.warning(f"Removing failed tool from registry: {tool_name}")
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
            "disabled_tools": [t for t in available_tools if t not in enabled_tools],
            "tools": {
                tool_name: cls.get_tool_info(tool_name)
                for tool_name in available_tools
            }
        }


# Export public API
__all__ = [
    'ToolRegistry',
    'ToolFactory',
    'ToolDependencies'
]
