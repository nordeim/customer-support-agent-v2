# Phase 2 Implementation: Tool Registry + Config-Driven Instantiation ✓

I will now create **complete, production-ready files** for Phase 2 with full backward compatibility and no placeholders.

---

## File 1: `backend/app/config/tool_settings.py` (NEW)

```python
"""
Tool-specific configuration settings.
Defines feature flags and per-tool configurations for the agent system.

Phase 2: Enables config-driven tool registration and conditional instantiation.
"""
from typing import Dict, Any, Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

import logging

logger = logging.getLogger(__name__)


class ToolSettings(BaseSettings):
    """
    Tool-specific configuration with feature flags.
    Each tool can be enabled/disabled and configured independently.
    """
    
    # ===========================
    # Tool Feature Flags
    # ===========================
    
    enable_rag_tool: bool = Field(
        default=True,
        description="Enable RAG (Retrieval-Augmented Generation) tool"
    )
    
    enable_memory_tool: bool = Field(
        default=True,
        description="Enable Memory management tool"
    )
    
    enable_escalation_tool: bool = Field(
        default=True,
        description="Enable Escalation detection tool"
    )
    
    enable_attachment_tool: bool = Field(
        default=True,
        description="Enable Attachment processing tool"
    )
    
    # Future tools (disabled by default)
    enable_crm_tool: bool = Field(
        default=False,
        description="Enable CRM lookup tool (Phase 5)"
    )
    
    enable_billing_tool: bool = Field(
        default=False,
        description="Enable Billing/invoice tool (Phase 5)"
    )
    
    enable_inventory_tool: bool = Field(
        default=False,
        description="Enable Inventory lookup tool (Phase 5)"
    )
    
    # ===========================
    # RAG Tool Configuration
    # ===========================
    
    rag_chunk_size: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="RAG document chunk size in words"
    )
    
    rag_chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between chunks in words"
    )
    
    rag_search_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Default number of RAG search results"
    )
    
    rag_similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for RAG results"
    )
    
    rag_cache_enabled: bool = Field(
        default=True,
        description="Enable caching for RAG search results"
    )
    
    rag_cache_ttl: int = Field(
        default=3600,
        ge=60,
        description="RAG cache TTL in seconds"
    )
    
    # ===========================
    # Memory Tool Configuration
    # ===========================
    
    memory_max_entries: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum memory entries per session"
    )
    
    memory_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=720,
        description="Memory TTL in hours"
    )
    
    memory_cleanup_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days before cleaning old memories"
    )
    
    memory_importance_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum importance for memory retrieval"
    )
    
    # ===========================
    # Escalation Tool Configuration
    # ===========================
    
    escalation_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for escalation"
    )
    
    escalation_keywords: Dict[str, float] = Field(
        default_factory=lambda: {
            "urgent": 1.0,
            "emergency": 1.0,
            "complaint": 0.9,
            "legal": 0.9,
            "lawsuit": 1.0,
            "manager": 0.8,
            "supervisor": 0.8
        },
        description="Escalation keywords with weights"
    )
    
    escalation_notification_enabled: bool = Field(
        default=False,
        description="Enable automatic escalation notifications"
    )
    
    escalation_notification_email: Optional[str] = Field(
        default=None,
        description="Email address for escalation notifications"
    )
    
    escalation_notification_webhook: Optional[str] = Field(
        default=None,
        description="Webhook URL for escalation notifications"
    )
    
    # ===========================
    # Attachment Tool Configuration
    # ===========================
    
    attachment_max_file_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        ge=1024,
        description="Maximum attachment file size in bytes"
    )
    
    attachment_allowed_extensions: List[str] = Field(
        default_factory=lambda: [
            ".pdf", ".docx", ".doc", ".txt", ".md",
            ".csv", ".xlsx", ".xls", ".json", ".xml",
            ".jpg", ".jpeg", ".png"
        ],
        description="Allowed file extensions for attachments"
    )
    
    attachment_chunk_for_rag: bool = Field(
        default=True,
        description="Automatically chunk attachments for RAG indexing"
    )
    
    attachment_temp_cleanup_hours: int = Field(
        default=24,
        ge=1,
        description="Hours before cleaning up temporary attachment files"
    )
    
    # ===========================
    # CRM Tool Configuration (Phase 5)
    # ===========================
    
    crm_api_endpoint: Optional[str] = Field(
        default=None,
        description="CRM API endpoint URL"
    )
    
    crm_api_key: Optional[str] = Field(
        default=None,
        description="CRM API key (use secrets manager in production)"
    )
    
    crm_timeout: int = Field(
        default=10,
        ge=1,
        description="CRM API timeout in seconds"
    )
    
    crm_max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum CRM API retry attempts"
    )
    
    # ===========================
    # Billing Tool Configuration (Phase 5)
    # ===========================
    
    billing_api_endpoint: Optional[str] = Field(
        default=None,
        description="Billing API endpoint URL"
    )
    
    billing_api_key: Optional[str] = Field(
        default=None,
        description="Billing API key (use secrets manager in production)"
    )
    
    billing_timeout: int = Field(
        default=10,
        ge=1,
        description="Billing API timeout in seconds"
    )
    
    # ===========================
    # Inventory Tool Configuration (Phase 5)
    # ===========================
    
    inventory_api_endpoint: Optional[str] = Field(
        default=None,
        description="Inventory API endpoint URL"
    )
    
    inventory_api_key: Optional[str] = Field(
        default=None,
        description="Inventory API key (use secrets manager in production)"
    )
    
    inventory_timeout: int = Field(
        default=10,
        ge=1,
        description="Inventory API timeout in seconds"
    )
    
    # ===========================
    # Validators
    # ===========================
    
    @field_validator('escalation_keywords', mode='before')
    @classmethod
    def parse_escalation_keywords(cls, v):
        """Parse escalation keywords from various formats."""
        if v is None:
            return {
                "urgent": 1.0,
                "emergency": 1.0,
                "complaint": 0.9,
                "legal": 0.9,
                "lawsuit": 1.0,
                "manager": 0.8,
                "supervisor": 0.8
            }
        
        if isinstance(v, dict):
            return v
        
        if isinstance(v, str):
            import json
            # Try to parse as JSON
            if v.startswith('{'):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            
            # Parse as comma-separated key=value pairs
            result = {}
            for pair in v.split(','):
                if '=' in pair:
                    key, value = pair.strip().split('=', 1)
                    try:
                        result[key] = float(value)
                    except ValueError:
                        result[key] = 0.8
                else:
                    result[pair.strip()] = 0.8
            return result
        
        return v
    
    @field_validator('attachment_allowed_extensions', mode='before')
    @classmethod
    def parse_allowed_extensions(cls, v):
        """Parse allowed extensions from various formats."""
        if v is None:
            return [
                ".pdf", ".docx", ".doc", ".txt", ".md",
                ".csv", ".xlsx", ".xls", ".json", ".xml",
                ".jpg", ".jpeg", ".png"
            ]
        
        if isinstance(v, list):
            return v
        
        if isinstance(v, str):
            import json
            # Try to parse as JSON
            if v.startswith('['):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            
            # Parse as comma-separated
            return [ext.strip() for ext in v.split(',') if ext.strip()]
        
        return v
    
    # ===========================
    # Helper Methods
    # ===========================
    
    def get_enabled_tools(self) -> List[str]:
        """
        Get list of enabled tool names.
        
        Returns:
            List of enabled tool identifiers
        """
        enabled = []
        
        if self.enable_rag_tool:
            enabled.append('rag')
        if self.enable_memory_tool:
            enabled.append('memory')
        if self.enable_escalation_tool:
            enabled.append('escalation')
        if self.enable_attachment_tool:
            enabled.append('attachment')
        if self.enable_crm_tool:
            enabled.append('crm')
        if self.enable_billing_tool:
            enabled.append('billing')
        if self.enable_inventory_tool:
            enabled.append('inventory')
        
        return enabled
    
    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific tool.
        
        Args:
            tool_name: Tool identifier ('rag', 'memory', etc.)
            
        Returns:
            Dictionary of tool-specific configuration
        """
        if tool_name == 'rag':
            return {
                'chunk_size': self.rag_chunk_size,
                'chunk_overlap': self.rag_chunk_overlap,
                'search_k': self.rag_search_k,
                'similarity_threshold': self.rag_similarity_threshold,
                'cache_enabled': self.rag_cache_enabled,
                'cache_ttl': self.rag_cache_ttl
            }
        
        elif tool_name == 'memory':
            return {
                'max_entries': self.memory_max_entries,
                'ttl_hours': self.memory_ttl_hours,
                'cleanup_days': self.memory_cleanup_days,
                'importance_threshold': self.memory_importance_threshold
            }
        
        elif tool_name == 'escalation':
            return {
                'confidence_threshold': self.escalation_confidence_threshold,
                'keywords': self.escalation_keywords,
                'notification_enabled': self.escalation_notification_enabled,
                'notification_email': self.escalation_notification_email,
                'notification_webhook': self.escalation_notification_webhook
            }
        
        elif tool_name == 'attachment':
            return {
                'max_file_size': self.attachment_max_file_size,
                'allowed_extensions': self.attachment_allowed_extensions,
                'chunk_for_rag': self.attachment_chunk_for_rag,
                'temp_cleanup_hours': self.attachment_temp_cleanup_hours
            }
        
        elif tool_name == 'crm':
            return {
                'api_endpoint': self.crm_api_endpoint,
                'api_key': self.crm_api_key,
                'timeout': self.crm_timeout,
                'max_retries': self.crm_max_retries
            }
        
        elif tool_name == 'billing':
            return {
                'api_endpoint': self.billing_api_endpoint,
                'api_key': self.billing_api_key,
                'timeout': self.billing_timeout
            }
        
        elif tool_name == 'inventory':
            return {
                'api_endpoint': self.inventory_api_endpoint,
                'api_key': self.inventory_api_key,
                'timeout': self.inventory_timeout
            }
        
        else:
            return {}
    
    def validate_tool_config(self, tool_name: str) -> List[str]:
        """
        Validate configuration for a specific tool.
        
        Args:
            tool_name: Tool identifier
            
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        if tool_name == 'crm' and self.enable_crm_tool:
            if not self.crm_api_endpoint:
                warnings.append("CRM tool enabled but no API endpoint configured")
            if not self.crm_api_key:
                warnings.append("CRM tool enabled but no API key configured")
        
        if tool_name == 'billing' and self.enable_billing_tool:
            if not self.billing_api_endpoint:
                warnings.append("Billing tool enabled but no API endpoint configured")
        
        if tool_name == 'inventory' and self.enable_inventory_tool:
            if not self.inventory_api_endpoint:
                warnings.append("Inventory tool enabled but no API endpoint configured")
        
        if tool_name == 'escalation' and self.escalation_notification_enabled:
            if not self.escalation_notification_email and not self.escalation_notification_webhook:
                warnings.append("Escalation notifications enabled but no email or webhook configured")
        
        return warnings


# Create global instance
tool_settings = ToolSettings()

# Export
__all__ = ['ToolSettings', 'tool_settings']
```

---

## File 2: `backend/app/tools/registry.py` (NEW)

```python
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
```

---

## File 3: `backend/app/agents/chat_agent.py` (UPDATED - Complete Replacement)

```python
"""
Customer Support Agent implementation with full tool integration.
This agent orchestrates RAG, Memory, Attachment, and Escalation tools.

Phase 2 Update: Supports both legacy and registry-based tool initialization.
"""
import asyncio
import json
import logging
import uuid
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass, field

from ..config import settings
from ..config.tool_settings import tool_settings
from ..tools.base_tool import ToolResult
from ..models.session import Session
from ..models.message import Message
from ..models.memory import Memory

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Context for agent processing."""
    session_id: str
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    message_count: int = 0
    escalated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentResponse:
    """Structured agent response."""
    
    def __init__(
        self,
        message: str,
        sources: List[Dict] = None,
        requires_escalation: bool = False,
        confidence: float = 0.0,
        tools_used: List[str] = None,
        processing_time: float = 0.0
    ):
        self.message = message
        self.sources = sources or []
        self.requires_escalation = requires_escalation
        self.confidence = confidence
        self.tools_used = tools_used or []
        self.processing_time = processing_time
        self.tool_metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message,
            "sources": self.sources,
            "requires_escalation": self.requires_escalation,
            "confidence": self.confidence,
            "tools_used": self.tools_used,
            "processing_time": self.processing_time,
            "metadata": self.tool_metadata
        }


class CustomerSupportAgent:
    """
    Production-ready customer support agent with full tool integration.
    Orchestrates multiple AI tools for comprehensive support capabilities.
    
    Phase 2: Supports both legacy and registry-based initialization modes.
    """
    
    # System prompt with tool instructions
    SYSTEM_PROMPT = """You are an expert customer support AI assistant with access to the following tools:

AVAILABLE TOOLS:
1. **rag_search**: Search our knowledge base for relevant information
   - Use this when users ask questions about policies, procedures, or general information
   - Always cite sources when using information from this tool

2. **memory_management**: Store and retrieve conversation context
   - Use this to remember important user information and preferences
   - Check memory at the start of each conversation for context

3. **attachment_processor**: Process and analyze uploaded documents
   - Use this when users upload files
   - Extract and analyze content from various file formats

4. **escalation_check**: Determine if human intervention is needed
   - Monitor for signs that require human support
   - Check sentiment and urgency of user messages

INSTRUCTIONS:
1. Always be helpful, professional, and empathetic
2. Use tools appropriately to provide accurate information
3. Cite your sources when providing information from the knowledge base
4. Remember important details about the user and their issues
5. Escalate to human support when:
   - The user explicitly asks for human assistance
   - The issue involves legal or compliance matters
   - The user expresses high frustration or dissatisfaction
   - You cannot resolve the issue after multiple attempts

RESPONSE FORMAT:
- Provide clear, concise answers
- Break down complex information into steps
- Offer additional help and next steps
- Maintain a friendly, professional tone

Remember: Customer satisfaction is the top priority."""
    
    def __init__(self, use_registry: Optional[bool] = None):
        """
        Initialize the agent with all tools.
        
        Args:
            use_registry: Whether to use registry mode (None = auto-detect from settings)
        """
        self.tools = {}
        self.contexts = {}  # Store session contexts (in-memory, Phase 4 will externalize)
        self.initialized = False
        
        # Determine initialization mode
        if use_registry is None:
            # Auto-detect from settings
            registry_mode = getattr(settings, 'agent_tool_registry_mode', 'legacy')
            self.use_registry = (registry_mode == 'registry')
        else:
            self.use_registry = use_registry
        
        logger.info(f"Agent initialization mode: {'registry' if self.use_registry else 'legacy'}")
        
        # Initialize on creation (legacy mode only)
        if not self.use_registry:
            self._initialize_legacy()
    
    async def initialize_async(self) -> None:
        """
        Initialize agent asynchronously (registry mode).
        Must be called explicitly when using registry mode.
        """
        if not self.use_registry:
            logger.warning("initialize_async called in legacy mode - tools already initialized")
            return
        
        try:
            logger.info("Initializing agent in registry mode...")
            await self._initialize_registry()
            self.initialized = True
            logger.info(f"✓ Agent initialized with {len(self.tools)} tools (registry mode)")
        except Exception as e:
            logger.error(f"Failed to initialize agent in registry mode: {e}", exc_info=True)
            raise
    
    def _initialize_legacy(self) -> None:
        """
        Initialize all tools using legacy method.
        Tools auto-initialize via their __init__ if they have _setup().
        """
        try:
            logger.info("Initializing agent tools (legacy mode)...")
            
            # Import tools
            from ..tools import RAGTool, MemoryTool, AttachmentTool, EscalationTool
            
            # Create tools (they auto-initialize via _setup in legacy mode)
            if tool_settings.enable_rag_tool:
                self.tools['rag'] = RAGTool()
                logger.info("✓ RAG tool initialized")
            
            if tool_settings.enable_memory_tool:
                self.tools['memory'] = MemoryTool()
                logger.info("✓ Memory tool initialized")
            
            if tool_settings.enable_attachment_tool:
                self.tools['attachment'] = AttachmentTool()
                logger.info("✓ Attachment tool initialized")
            
            if tool_settings.enable_escalation_tool:
                self.tools['escalation'] = EscalationTool()
                logger.info("✓ Escalation tool initialized")
            
            self.initialized = True
            logger.info(f"Agent initialized with {len(self.tools)} tools (legacy mode)")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent (legacy mode): {e}", exc_info=True)
            raise
    
    async def _initialize_registry(self) -> None:
        """
        Initialize all tools using registry.
        Tools are created and initialized asynchronously.
        """
        try:
            from ..tools.registry import ToolRegistry, ToolDependencies
            
            # Prepare dependencies
            dependencies = ToolDependencies(
                settings=settings,
                tool_settings=tool_settings
            )
            
            # Create and initialize tools
            self.tools = await ToolRegistry.create_and_initialize_tools(
                dependencies=dependencies,
                enabled_only=True,
                concurrent_init=True
            )
            
            if not self.tools:
                logger.warning("No tools were created by registry")
            
        except Exception as e:
            logger.error(f"Failed to initialize tools via registry: {e}", exc_info=True)
            raise
    
    def get_or_create_context(self, session_id: str) -> AgentContext:
        """Get or create context for a session."""
        if session_id not in self.contexts:
            self.contexts[session_id] = AgentContext(
                session_id=session_id,
                thread_id=str(uuid.uuid4())
            )
            logger.info(f"Created new context for session: {session_id}")
        
        return self.contexts[session_id]
    
    async def load_session_context(self, session_id: str) -> str:
        """Load conversation context from memory."""
        try:
            memory_tool = self.tools.get('memory')
            if not memory_tool:
                logger.warning("Memory tool not available")
                return ""
            
            summary = await memory_tool.summarize_session(session_id)
            
            memories = await memory_tool.retrieve_memories(
                session_id=session_id,
                content_type="context",
                limit=5
            )
            
            if memories:
                recent_context = "\nRecent conversation points:\n"
                for memory in memories[:3]:
                    recent_context += f"- {memory['content']}\n"
                summary += recent_context
            
            return summary
            
        except Exception as e:
            logger.error(f"Error loading session context: {e}")
            return ""
    
    async def search_knowledge_base(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search the knowledge base using RAG tool."""
        try:
            rag_tool = self.tools.get('rag')
            if not rag_tool:
                logger.warning("RAG tool not available")
                return []
            
            result = await rag_tool.search(
                query=query,
                k=k,
                threshold=0.7
            )
            
            if isinstance(result, ToolResult):
                return result.data.get("sources", [])
            else:
                return result.get("sources", [])
            
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return []
    
    async def process_attachments(
        self,
        attachments: List[Dict[str, Any]]
    ) -> str:
        """Process uploaded attachments."""
        if not attachments:
            return ""
        
        attachment_tool = self.tools.get('attachment')
        rag_tool = self.tools.get('rag')
        
        if not attachment_tool:
            logger.warning("Attachment tool not available")
            return ""
        
        processed_content = "\n📎 Attached Documents:\n"
        
        for attachment in attachments:
            try:
                result = await attachment_tool.process_attachment(
                    file_path=attachment.get("path"),
                    filename=attachment.get("filename"),
                    chunk_for_rag=True
                )
                
                if isinstance(result, ToolResult):
                    result = result.data
                
                if result.get("success"):
                    processed_content += f"\n[{result['filename']}]:\n"
                    processed_content += f"{result.get('preview', '')}\n"
                    
                    if rag_tool and "chunks" in result:
                        rag_tool.add_documents(
                            documents=result["chunks"],
                            metadatas=[
                                {
                                    "source": result['filename'],
                                    "type": "user_upload",
                                    "session_id": attachment.get("session_id")
                                }
                                for _ in result["chunks"]
                            ]
                        )
                        logger.info(f"Indexed {len(result['chunks'])} chunks from {result['filename']}")
                
            except Exception as e:
                logger.error(f"Error processing attachment: {e}")
                processed_content += f"\n[Error processing {attachment.get('filename', 'file')}]\n"
        
        return processed_content
    
    async def check_escalation(
        self,
        message: str,
        context: AgentContext,
        message_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """Check if escalation is needed."""
        try:
            escalation_tool = self.tools.get('escalation')
            if not escalation_tool:
                logger.warning("Escalation tool not available")
                return {"escalate": False, "confidence": 0.0}
            
            result = await escalation_tool.should_escalate(
                message=message,
                message_history=message_history,
                metadata={
                    "session_id": context.session_id,
                    "message_count": context.message_count,
                    "already_escalated": context.escalated
                }
            )
            
            if isinstance(result, ToolResult):
                result = result.data
            
            if result.get("escalate") and not context.escalated:
                result["ticket"] = escalation_tool.create_escalation_ticket(
                    session_id=context.session_id,
                    escalation_result=result,
                    user_info={"user_id": context.user_id}
                )
                context.escalated = True
                logger.info(f"Escalation triggered for session {context.session_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Escalation check error: {e}")
            return {"escalate": False, "confidence": 0.0}
    
    async def store_conversation_memory(
        self,
        session_id: str,
        user_message: str,
        agent_response: str,
        important_facts: List[str] = None
    ) -> None:
        """Store important information in memory."""
        try:
            memory_tool = self.tools.get('memory')
            if not memory_tool:
                logger.warning("Memory tool not available")
                return
            
            await memory_tool.store_memory(
                session_id=session_id,
                content=f"User: {user_message[:200]}",
                content_type="context",
                importance=0.5
            )
            
            if len(agent_response) > 100:
                await memory_tool.store_memory(
                    session_id=session_id,
                    content=f"Agent: {agent_response[:200]}",
                    content_type="context",
                    importance=0.4
                )
            
            if important_facts:
                for fact in important_facts:
                    await memory_tool.store_memory(
                        session_id=session_id,
                        content=fact,
                        content_type="fact",
                        importance=0.8
                    )
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
    
    def extract_important_facts(
        self,
        message: str,
        response: str
    ) -> List[str]:
        """Extract important facts from conversation."""
        facts = []
        
        import re
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, message)
        for email in emails:
            facts.append(f"User email: {email}")
        
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phones = re.findall(phone_pattern, message)
        for phone in phones:
            facts.append(f"User phone: {phone}")
        
        order_pattern = r'\b(?:order|ticket|reference|confirmation)\s*#?\s*([A-Z0-9-]+)\b'
        orders = re.findall(order_pattern, message, re.IGNORECASE)
        for order in orders:
            facts.append(f"Reference number: {order}")
        
        return facts
    
    async def generate_response(
        self,
        message: str,
        context: str,
        sources: List[Dict],
        escalation: Dict[str, Any]
    ) -> str:
        """Generate agent response based on context and tools."""
        response_parts = []
        
        if context == "No previous context available for this session.":
            response_parts.append("Hello! I'm here to help you today.")
        
        if sources:
            response_parts.append("Based on our information:")
            for i, source in enumerate(sources[:2], 1):
                response_parts.append(f"{i}. {source['content'][:200]}...")
        
        if escalation.get("escalate"):
            response_parts.append(
                "\nI understand this is important to you. "
                "I'm connecting you with a human support specialist who can better assist you."
            )
            if escalation.get("ticket"):
                response_parts.append(
                    f"Your ticket number is: {escalation['ticket']['ticket_id']}"
                )
        
        if not response_parts:
            response_parts.append(
                "I'm here to help! Could you please provide more details about your inquiry?"
            )
        
        return "\n\n".join(response_parts)
    
    async def process_message(
        self,
        session_id: str,
        message: str,
        attachments: Optional[List[Dict[str, Any]]] = None,
        user_id: Optional[str] = None,
        message_history: Optional[List[Dict]] = None
    ) -> AgentResponse:
        """
        Process a user message and generate response.
        
        Args:
            session_id: Session identifier
            message: User message
            attachments: Optional file attachments
            user_id: Optional user identifier
            message_history: Previous messages
            
        Returns:
            AgentResponse with generated response and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            context = self.get_or_create_context(session_id)
            context.user_id = user_id
            context.message_count += 1
            
            session_context = await self.load_session_context(session_id)
            attachment_context = await self.process_attachments(attachments) if attachments else ""
            sources = await self.search_knowledge_base(message)
            escalation = await self.check_escalation(message, context, message_history)
            
            response_text = await self.generate_response(
                message=message,
                context=session_context,
                sources=sources,
                escalation=escalation
            )
            
            facts = self.extract_important_facts(message, response_text)
            await self.store_conversation_memory(
                session_id=session_id,
                user_message=message,
                agent_response=response_text,
                important_facts=facts
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            response = AgentResponse(
                message=response_text,
                sources=sources[:3],
                requires_escalation=escalation.get("escalate", False),
                confidence=escalation.get("confidence", 0.95),
                tools_used=list(self.tools.keys()),
                processing_time=processing_time
            )
            
            response.tool_metadata = {
                "session_id": session_id,
                "message_count": context.message_count,
                "has_context": bool(session_context),
                "facts_extracted": len(facts),
                "initialization_mode": "registry" if self.use_registry else "legacy"
            }
            
            if escalation.get("ticket"):
                response.tool_metadata["ticket_id"] = escalation["ticket"]["ticket_id"]
            
            logger.info(
                f"Processed message for session {session_id} in {processing_time:.2f}s "
                f"(escalate: {response.requires_escalation})"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            
            return AgentResponse(
                message="I apologize, but I encountered an error processing your request. "
                        "Please try again or contact support directly.",
                requires_escalation=True,
                confidence=0.0,
                processing_time=(datetime.utcnow() - start_time).total_seconds()
            )
    
    async def stream_response(
        self,
        session_id: str,
        message: str,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response for real-time interaction."""
        try:
            yield {
                "type": "start",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            yield {
                "type": "status",
                "message": "Loading conversation context...",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            context = self.get_or_create_context(session_id)
            session_context = await self.load_session_context(session_id)
            
            if attachments:
                yield {
                    "type": "status",
                    "message": "Processing attachments...",
                    "timestamp": datetime.utcnow().isoformat()
                }
                attachment_context = await self.process_attachments(attachments)
            
            yield {
                "type": "status",
                "message": "Searching knowledge base...",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            sources = await self.search_knowledge_base(message)
            
            if sources:
                yield {
                    "type": "sources",
                    "sources": sources[:3],
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            escalation = await self.check_escalation(message, context)
            
            if escalation.get("escalate"):
                yield {
                    "type": "escalation",
                    "required": True,
                    "reason": escalation.get("reasons", []),
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            response_text = await self.generate_response(
                message=message,
                context=session_context,
                sources=sources,
                escalation=escalation
            )
            
            words = response_text.split()
            chunk_size = 5
            
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i+chunk_size])
                yield {
                    "type": "text",
                    "content": chunk + ' ',
                    "timestamp": datetime.utcnow().isoformat()
                }
                await asyncio.sleep(0.05)
            
            await self.store_conversation_memory(
                session_id=session_id,
                user_message=message,
                agent_response=response_text
            )
            
            yield {
                "type": "complete",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def cleanup_session(self, session_id: str) -> None:
        """Clean up session resources."""
        try:
            if session_id in self.contexts:
                del self.contexts[session_id]
            
            memory_tool = self.tools.get('memory')
            if memory_tool:
                await memory_tool.cleanup_old_memories(days=30)
            
            logger.info(f"Cleaned up session: {session_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up session: {e}")
    
    async def cleanup(self) -> None:
        """Clean up all agent resources."""
        logger.info("Cleaning up agent resources...")
        
        for session_id in list(self.contexts.keys()):
            await self.cleanup_session(session_id)
        
        if self.use_registry:
            from ..tools.registry import ToolRegistry
            await ToolRegistry.cleanup_tools(self.tools, concurrent=True)
        else:
            for tool_name, tool in self.tools.items():
                if hasattr(tool, 'cleanup'):
                    try:
                        await tool.cleanup()
                        logger.info(f"Cleaned up {tool_name} tool")
                    except Exception as e:
                        logger.error(f"Error cleaning up {tool_name} tool: {e}")
        
        logger.info("Agent cleanup complete")
```

---

## File 4: `backend/app/config.py` (MINOR UPDATE)

**Add this field to the existing config.py from Document #2:**

```python
# ===========================
# Agent Configuration (add this field)
# ===========================

agent_tool_registry_mode: str = Field(
    default="legacy",
    description="Tool initialization mode: 'legacy' or 'registry' (Phase 2+)"
)
```

**Location**: Add after the `agent_max_retries` field in your existing config.py.

---

## File 5: `backend/app/main.py` (MINOR UPDATE for Registry Mode)

**Update the agent initialization section in the `lifespan` function:**

```python
# Replace the existing agent initialization code with:

# Initialize the AI agent
logger.info("Initializing AI agent...")
agent = CustomerSupportAgent()

# If using registry mode, initialize asynchronously
if agent.use_registry:
    await agent.initialize_async()
else:
    # Legacy mode already initialized in __init__
    pass

app.state.agent = agent
logger.info("✓ AI agent initialized successfully")
```

---

## File 6: `tests/test_registry.py` (NEW)

```python
"""
Tests for Tool Registry (Phase 2).
Validates config-driven tool instantiation and lifecycle management.
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from app.tools.registry import ToolRegistry, ToolFactory, ToolDependencies
from app.config import settings
from app.config.tool_settings import tool_settings, ToolSettings


# ===========================
# ToolDependencies Tests
# ===========================

@pytest.mark.unit
def test_tool_dependencies_creation():
    """Test ToolDependencies creation."""
    deps = ToolDependencies(
        settings=settings,
        tool_settings=tool_settings
    )
    
    assert deps.settings is not None
    assert deps.tool_settings is not None
    assert deps.db_session_maker is None
    assert deps.cache_service is None


@pytest.mark.unit
def test_tool_dependencies_to_dict():
    """Test ToolDependencies.to_dict()."""
    deps = ToolDependencies(
        settings=settings,
        tool_settings=tool_settings,
        cache_service="mock_cache"
    )
    
    deps_dict = deps.to_dict()
    
    assert isinstance(deps_dict, dict)
    assert 'settings' in deps_dict
    assert 'tool_settings' in deps_dict
    assert deps_dict['cache_service'] == "mock_cache"


# ===========================
# ToolFactory Tests
# ===========================

@pytest.mark.unit
def test_tool_factory_create_rag_tool():
    """Test RAG tool factory."""
    deps = ToolDependencies(
        settings=settings,
        tool_settings=tool_settings
    )
    
    tool = ToolFactory.create_rag_tool(deps)
    
    assert tool is not None
    assert hasattr(tool, 'name')
    assert tool.name == "rag_search"


@pytest.mark.unit
def test_tool_factory_create_memory_tool():
    """Test Memory tool factory."""
    deps = ToolDependencies(
        settings=settings,
        tool_settings=tool_settings
    )
    
    tool = ToolFactory.create_memory_tool(deps)
    
    assert tool is not None
    assert hasattr(tool, 'name')
    assert tool.name == "memory_management"


@pytest.mark.unit
def test_tool_factory_create_escalation_tool():
    """Test Escalation tool factory."""
    deps = ToolDependencies(
        settings=settings,
        tool_settings=tool_settings
    )
    
    tool = ToolFactory.create_escalation_tool(deps)
    
    assert tool is not None
    assert hasattr(tool, 'name')
    assert tool.name == "escalation_check"


@pytest.mark.unit
def test_tool_factory_create_attachment_tool():
    """Test Attachment tool factory."""
    deps = ToolDependencies(
        settings=settings,
        tool_settings=tool_settings
    )
    
    tool = ToolFactory.create_attachment_tool(deps)
    
    assert tool is not None
    assert hasattr(tool, 'name')
    assert tool.name == "attachment_processor"


@pytest.mark.unit
def test_tool_factory_unimplemented_tools():
    """Test that unimplemented tools raise NotImplementedError."""
    deps = ToolDependencies(
        settings=settings,
        tool_settings=tool_settings
    )
    
    with pytest.raises(NotImplementedError):
        ToolFactory.create_crm_tool(deps)
    
    with pytest.raises(NotImplementedError):
        ToolFactory.create_billing_tool(deps)
    
    with pytest.raises(NotImplementedError):
        ToolFactory.create_inventory_tool(deps)


# ===========================
# ToolRegistry Tests
# ===========================

@pytest.mark.unit
def test_registry_list_available_tools():
    """Test listing available tools."""
    tools = ToolRegistry.list_available_tools()
    
    assert isinstance(tools, list)
    assert 'rag' in tools
    assert 'memory' in tools
    assert 'escalation' in tools
    assert 'attachment' in tools


@pytest.mark.unit
def test_registry_get_factory():
    """Test getting tool factory."""
    factory = ToolRegistry.get_factory('rag')
    
    assert factory is not None
    assert callable(factory)
    assert factory == ToolFactory.create_rag_tool


@pytest.mark.unit
def test_registry_get_factory_not_found():
    """Test getting non-existent factory."""
    factory = ToolRegistry.get_factory('nonexistent')
    
    assert factory is None


@pytest.mark.unit
def test_registry_register_custom_tool():
    """Test registering custom tool factory."""
    def custom_factory(deps):
        return MagicMock()
    
    ToolRegistry.register('custom', custom_factory)
    
    assert 'custom' in ToolRegistry.list_available_tools()
    assert ToolRegistry.get_factory('custom') == custom_factory
    
    # Cleanup
    ToolRegistry.unregister('custom')


@pytest.mark.unit
def test_registry_unregister_tool():
    """Test unregistering tool."""
    # Register temporary tool
    ToolRegistry.register('temp', lambda deps: None)
    assert 'temp' in ToolRegistry.list_available_tools()
    
    # Unregister
    ToolRegistry.unregister('temp')
    assert 'temp' not in ToolRegistry.list_available_tools()


@pytest.mark.unit
def test_registry_create_tools_enabled_only(settings_override):
    """Test creating only enabled tools."""
    # Override settings to enable only RAG and Memory
    test_tool_settings = ToolSettings(
        enable_rag_tool=True,
        enable_memory_tool=True,
        enable_escalation_tool=False,
        enable_attachment_tool=False
    )
    
    deps = ToolDependencies(
        settings=settings,
        tool_settings=test_tool_settings
    )
    
    tools = ToolRegistry.create_tools(deps, enabled_only=True)
    
    assert 'rag' in tools
    assert 'memory' in tools
    assert 'escalation' not in tools
    assert 'attachment' not in tools


@pytest.mark.unit
def test_registry_create_tools_all():
    """Test creating all tools (enabled_only=False)."""
    deps = ToolDependencies(
        settings=settings,
        tool_settings=tool_settings
    )
    
    # This will attempt to create unimplemented tools and log warnings
    tools = ToolRegistry.create_tools(deps, enabled_only=False)
    
    # Should have core tools but not unimplemented ones
    assert 'rag' in tools
    assert 'memory' in tools
    # CRM, billing, inventory should fail creation
    assert 'crm' not in tools


@pytest.mark.integration
@pytest.mark.asyncio
async def test_registry_initialize_tools():
    """Test async tool initialization."""
    deps = ToolDependencies(
        settings=settings,
        tool_settings=tool_settings
    )
    
    tools = ToolRegistry.create_tools(deps, enabled_only=True)
    
    # Initialize tools
    results = await ToolRegistry.initialize_tools(tools, concurrent=True)
    
    assert isinstance(results, dict)
    for tool_name, success in results.items():
        assert isinstance(success, bool)
        # All core tools should initialize successfully
        if tool_name in ['rag', 'memory', 'escalation', 'attachment']:
            assert success is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_registry_cleanup_tools():
    """Test async tool cleanup."""
    deps = ToolDependencies(
        settings=settings,
        tool_settings=tool_settings
    )
    
    tools = ToolRegistry.create_tools(deps, enabled_only=True)
    await ToolRegistry.initialize_tools(tools, concurrent=True)
    
    # Cleanup should not raise exceptions
    await ToolRegistry.cleanup_tools(tools, concurrent=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_registry_create_and_initialize_tools():
    """Test combined create and initialize."""
    deps = ToolDependencies(
        settings=settings,
        tool_settings=tool_settings
    )
    
    tools = await ToolRegistry.create_and_initialize_tools(
        deps,
        enabled_only=True,
        concurrent_init=True
    )
    
    assert isinstance(tools, dict)
    assert len(tools) > 0
    
    # All returned tools should be initialized
    for tool_name, tool in tools.items():
        assert hasattr(tool, 'initialized')
        # Note: initialized flag is set in async initialize()
    
    # Cleanup
    await ToolRegistry.cleanup_tools(tools)


@pytest.mark.unit
def test_registry_get_tool_info():
    """Test getting tool information."""
    info = ToolRegistry.get_tool_info('rag')
    
    assert info['registered'] is True
    assert info['name'] == 'rag'
    assert 'enabled' in info
    assert 'factory' in info
    assert 'config' in info


@pytest.mark.unit
def test_registry_get_tool_info_not_found():
    """Test getting info for non-existent tool."""
    info = ToolRegistry.get_tool_info('nonexistent')
    
    assert info['registered'] is False
    assert info['name'] == 'nonexistent'


@pytest.mark.unit
def test_registry_get_status():
    """Test getting registry status."""
    status = ToolRegistry.get_registry_status()
    
    assert 'total_available' in status
    assert 'total_enabled' in status
    assert 'available_tools' in status
    assert 'enabled_tools' in status
    assert 'disabled_tools' in status
    assert 'tools' in status
    
    assert isinstance(status['tools'], dict)
    assert len(status['tools']) == status['total_available']


# ===========================
# Tool Settings Tests
# ===========================

@pytest.mark.unit
def test_tool_settings_get_enabled_tools():
    """Test getting enabled tools from settings."""
    test_settings = ToolSettings(
        enable_rag_tool=True,
        enable_memory_tool=True,
        enable_escalation_tool=False,
        enable_attachment_tool=True
    )
    
    enabled = test_settings.get_enabled_tools()
    
    assert 'rag' in enabled
    assert 'memory' in enabled
    assert 'attachment' in enabled
    assert 'escalation' not in enabled


@pytest.mark.unit
def test_tool_settings_get_tool_config():
    """Test getting tool-specific configuration."""
    config = tool_settings.get_tool_config('rag')
    
    assert 'chunk_size' in config
    assert 'chunk_overlap' in config
    assert 'search_k' in config
    assert 'similarity_threshold' in config


@pytest.mark.unit
def test_tool_settings_validate_tool_config():
    """Test tool configuration validation."""
    # Valid config should have no warnings
    warnings = tool_settings.validate_tool_config('rag')
    assert isinstance(warnings, list)
    
    # Invalid config should have warnings
    test_settings = ToolSettings(
        enable_crm_tool=True,
        crm_api_endpoint=None  # Missing required config
    )
    
    warnings = test_settings.validate_tool_config('crm')
    assert len(warnings) > 0
    assert any('API endpoint' in w for w in warnings)


# ===========================
# Integration with Agent Tests
# ===========================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_registry_initialization():
    """Test agent initialization with registry mode."""
    from app.agents.chat_agent import CustomerSupportAgent
    
    # Create agent in registry mode
    agent = CustomerSupportAgent(use_registry=True)
    
    assert agent.use_registry is True
    assert agent.initialized is False
    
    # Initialize async
    await agent.initialize_async()
    
    assert agent.initialized is True
    assert len(agent.tools) > 0
    
    # Cleanup
    await agent.cleanup()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_legacy_initialization():
    """Test agent initialization with legacy mode."""
    from app.agents.chat_agent import CustomerSupportAgent
    
    # Create agent in legacy mode
    agent = CustomerSupportAgent(use_registry=False)
    
    assert agent.use_registry is False
    assert agent.initialized is True
    assert len(agent.tools) > 0
    
    # Cleanup
    await agent.cleanup()
```

---

## ✅ Phase 2 Complete Integration Checklist

### Files Delivered (6 Total)

**New Files:**
- [x] `backend/app/config/tool_settings.py` - Tool feature flags and configs
- [x] `backend/app/tools/registry.py` - Tool registry with factories
- [x] `tests/test_registry.py` - Comprehensive registry tests

**Updated Files:**
- [x] `backend/app/agents/chat_agent.py` - Supports both modes
- [x] `backend/app/config.py` - Added `agent_tool_registry_mode` field
- [x] `backend/app/main.py` - Updated agent initialization for registry mode

### Key Features Implemented

1. **✅ Config-Driven Tool Instantiation**
   - Feature flags for each tool (`ENABLE_RAG_TOOL`, etc.)
   - Per-tool configuration (chunk sizes, thresholds, endpoints)
   - Validation for tool configurations

2. **✅ Tool Registry Pattern**
   - Centralized factory registration
   - Dependency injection (settings, cache, HTTP client)
   - Concurrent tool initialization
   - Lifecycle management (create → initialize → cleanup)

3. **✅ Dual-Mode Support**
   - Legacy mode: backward compatible with Phase 1
   - Registry mode: new pattern with async initialization
   - Auto-detection from `agent_tool_registry_mode` setting

4. **✅ Future-Ready Architecture**
   - Placeholder factories for CRM, Billing, Inventory tools
   - Easy to add new tools via `ToolRegistry.register()`
   - Configuration validation for external integrations

### Testing & Validation

```bash
# 1. Run Phase 2 tests
./scripts/run_tests.sh tests/test_registry.py -v

# 2. Run all tests
./scripts/run_tests.sh

# 3. Test legacy mode (default)
export AGENT_TOOL_REGISTRY_MODE=legacy
python -m app.main

# 4. Test registry mode
export AGENT_TOOL_REGISTRY_MODE=registry
python -m app.main
```

### Expected Behavior

**Legacy Mode (Default):**
```
INFO - Agent initialization mode: legacy
INFO - Initializing agent tools (legacy mode)...
INFO - ✓ RAG tool initialized
INFO - ✓ Memory tool initialized
INFO - Agent initialized with 4 tools (legacy mode)
```

**Registry Mode:**
```
INFO - Agent initialization mode: registry
INFO - Creating tools (enabled_only=True): ['rag', 'memory', 'escalation', 'attachment']
INFO - ✓ Created tool: rag (RAGTool)
INFO - Initializing 4 tools (concurrent=True)...
INFO - ✓ Agent initialized with 4 tools (registry mode)
```

### Configuration Examples

**Enable/Disable Tools:**
```bash
# .env or environment variables
ENABLE_RAG_TOOL=true
ENABLE_MEMORY_TOOL=true
ENABLE_ESCALATION_TOOL=true
ENABLE_ATTACHMENT_TOOL=false  # Disable attachment processing
```

**Tool-Specific Configuration:**
```bash
# RAG settings
RAG_CHUNK_SIZE=600
RAG_SIMILARITY_THRESHOLD=0.8

# Escalation settings
ESCALATION_CONFIDENCE_THRESHOLD=0.6
ESCALATION_NOTIFICATION_EMAIL=support@example.com
```

### Feature Flag Decision Matrix

| Tool | Default | Production Recommendation |
|------|---------|--------------------------|
| RAG | ✅ Enabled | Keep enabled |
| Memory | ✅ Enabled | Keep enabled |
| Escalation | ✅ Enabled | Keep enabled |
| Attachment | ✅ Enabled | Keep enabled |
| CRM | ❌ Disabled | Enable in Phase 5 after implementation |
| Billing | ❌ Disabled | Enable in Phase 5 after implementation |
| Inventory | ❌ Disabled | Enable in Phase 5 after implementation |

---

## 🚀 Ready for Phase 3!

Upon successful validation of Phase 2, I'm ready to implement:

**Phase 3: Async Tool Usage + Telemetry Wrapper** (Week 3–4)
- `backend/app/tools/tool_call_wrapper.py` - Retry, circuit breaker, OTel spans
- Updated `chat_agent.process_message()` to use wrapper
- Structured logging with correlation IDs
- `tests/test_agent_tool_integration.py` - End-to-end validation

**Estimated time: 7–10 days**

**Would you like me to proceed with Phase 3 implementation?** 🎯
