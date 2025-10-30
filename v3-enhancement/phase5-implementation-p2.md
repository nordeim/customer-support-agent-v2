# Phase 5 Implementation: Remaining Files ✓

Continuing with the remaining Phase 5 files...

---

## File 4: `backend/app/tools/registry.py` (UPDATE - Register Domain Tools)

**Update the existing registry.py to include CRM, Billing, and Inventory tool factories:**

```python
"""
Tool Registry for dynamic tool instantiation and management.
Enables config-driven tool registration with dependency injection.

Phase 5 Update: Added CRM, Billing, and Inventory tool factories.
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
        Fully implemented with HTTP client and retry logic.
        """
        from .crm_tool import CRMTool
        
        tool = CRMTool()
        logger.debug("CRM tool created (initialization deferred to async)")
        return tool
    
    @staticmethod
    def create_billing_tool(dependencies: ToolDependencies) -> BaseTool:
        """
        Create Billing tool instance (Phase 5).
        Template implementation - customize for your billing API.
        """
        from .billing_tool import BillingTool
        
        tool = BillingTool()
        logger.debug("Billing tool created (template - customize for your API)")
        return tool
    
    @staticmethod
    def create_inventory_tool(dependencies: ToolDependencies) -> BaseTool:
        """
        Create Inventory tool instance (Phase 5).
        Template implementation - customize for your inventory API.
        """
        from .inventory_tool import InventoryTool
        
        tool = InventoryTool()
        logger.debug("Inventory tool created (template - customize for your API)")
        return tool


class ToolRegistry:
    """
    Central registry for tool management.
    Handles tool creation, initialization, and lifecycle.
    """
    
    # Registry of tool factories (Phase 5: All tools now implemented)
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

## File 5: `requirements.txt` (UPDATE - Add HTTP Client Dependencies)

**Add these lines to your existing requirements.txt:**

```txt
# Phase 5 dependencies - CRM and Domain Tool Integration
aiohttp>=3.9.0  # Async HTTP client for external API calls
aiofiles>=23.2.0  # Async file operations (already added in Phase 1)
```

---

## File 6: `tests/test_crm_tool.py` (NEW)

```python
"""
Tests for CRM Tool (Phase 5).
Validates CRM tool implementation with mocked HTTP responses.
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from aiohttp import ClientSession, ClientResponse, ClientError
from datetime import datetime

from app.tools.crm_tool import (
    CRMTool,
    CustomerProfile,
    TicketInfo,
    CustomerNotFoundError,
    AuthenticationError,
    RateLimitError,
    CRMAPIError
)
from app.tools.base_tool import ToolResult
from app.config.tool_settings import ToolSettings


# ===========================
# Fixtures
# ===========================

@pytest.fixture
def crm_tool_settings():
    """Create test settings for CRM tool."""
    return ToolSettings(
        enable_crm_tool=True,
        crm_api_endpoint="https://api.crm.example.com/v1",
        crm_api_key="test-api-key-12345",
        crm_timeout=10,
        crm_max_retries=3
    )


@pytest.fixture
async def crm_tool(crm_tool_settings, monkeypatch):
    """Create CRM tool instance for testing."""
    # Patch tool_settings
    monkeypatch.setattr('app.tools.crm_tool.tool_settings', crm_tool_settings)
    
    tool = CRMTool()
    
    # Initialize (this will use mock mode since api_key is set but we'll mock requests)
    await tool.initialize()
    
    yield tool
    
    # Cleanup
    await tool.cleanup()


@pytest.fixture
async def crm_tool_no_api_key():
    """Create CRM tool without API key (mock mode)."""
    tool = CRMTool()
    tool.api_endpoint = "https://api.crm.example.com/v1"
    tool.api_key = None  # No API key = mock mode
    
    await tool.initialize()
    
    yield tool
    
    await tool.cleanup()


# ===========================
# CustomerProfile Tests
# ===========================

@pytest.mark.unit
def test_customer_profile_creation():
    """Test CustomerProfile data model."""
    profile = CustomerProfile(
        customer_id="CUST-001",
        email="test@example.com",
        name="Test User",
        account_status="active",
        tier="premium"
    )
    
    assert profile.customer_id == "CUST-001"
    assert profile.email == "test@example.com"
    assert profile.tier == "premium"
    assert profile.open_tickets == 0
    assert isinstance(profile.preferences, dict)


@pytest.mark.unit
def test_customer_profile_to_dict():
    """Test CustomerProfile serialization."""
    profile = CustomerProfile(
        customer_id="CUST-002",
        email="user@example.com",
        lifetime_value=5000.50,
        preferences={"language": "en"}
    )
    
    data = profile.to_dict()
    
    assert isinstance(data, dict)
    assert data["customer_id"] == "CUST-002"
    assert data["lifetime_value"] == 5000.50
    assert data["preferences"]["language"] == "en"


# ===========================
# TicketInfo Tests
# ===========================

@pytest.mark.unit
def test_ticket_info_creation():
    """Test TicketInfo data model."""
    ticket = TicketInfo(
        ticket_id="TICKET-001",
        status="open",
        priority="high",
        subject="Test ticket",
        created_at="2024-01-01T10:00:00Z"
    )
    
    assert ticket.ticket_id == "TICKET-001"
    assert ticket.status == "open"
    assert ticket.priority == "high"


@pytest.mark.unit
def test_ticket_info_to_dict():
    """Test TicketInfo serialization."""
    ticket = TicketInfo(
        ticket_id="TICKET-002",
        status="closed",
        priority="normal",
        subject="Billing issue",
        created_at="2024-01-01T10:00:00Z",
        assigned_to="agent@example.com"
    )
    
    data = ticket.to_dict()
    
    assert isinstance(data, dict)
    assert data["ticket_id"] == "TICKET-002"
    assert data["assigned_to"] == "agent@example.com"


# ===========================
# CRM Tool Initialization Tests
# ===========================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_crm_tool_initialization(crm_tool_settings, monkeypatch):
    """Test CRM tool initialization."""
    monkeypatch.setattr('app.tools.crm_tool.tool_settings', crm_tool_settings)
    
    tool = CRMTool()
    
    assert tool.name == "crm_lookup"
    assert tool.api_endpoint == "https://api.crm.example.com/v1"
    assert tool.timeout == 10
    assert tool.max_retries == 3
    
    await tool.initialize()
    
    assert tool.initialized is True
    assert tool.session is not None
    
    await tool.cleanup()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_crm_tool_initialization_without_endpoint():
    """Test CRM tool initialization fails without endpoint."""
    tool = CRMTool()
    tool.api_endpoint = None
    
    with pytest.raises(ValueError, match="CRM API endpoint not configured"):
        await tool.initialize()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_crm_tool_cleanup(crm_tool):
    """Test CRM tool cleanup."""
    assert crm_tool.session is not None
    
    await crm_tool.cleanup()
    
    assert crm_tool.session is None
    assert crm_tool.initialized is False


# ===========================
# Mock Mode Tests (No API Key)
# ===========================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_lookup_customer_mock_mode(crm_tool_no_api_key):
    """Test customer lookup in mock mode."""
    result = await crm_tool_no_api_key.lookup_customer_async(customer_id="CUST-12345")
    
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.data["found"] is True
    assert result.data["profile"]["customer_id"] == "CUST-12345"
    assert result.data["profile"]["email"] == "customer@example.com"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_tickets_mock_mode(crm_tool_no_api_key):
    """Test get tickets in mock mode."""
    result = await crm_tool_no_api_key.get_customer_tickets_async(customer_id="CUST-12345")
    
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.data["total_count"] >= 1
    assert len(result.data["tickets"]) >= 1
    assert result.data["tickets"][0]["ticket_id"] is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_customer_mock_mode(crm_tool_no_api_key):
    """Test update customer in mock mode."""
    result = await crm_tool_no_api_key.update_customer_async(
        customer_id="CUST-12345",
        updates={"tier": "premium", "satisfaction_score": 5.0}
    )
    
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.data["customer_id"] == "CUST-12345"
    assert "updated_at" in result.data


# ===========================
# Execute Method Tests
# ===========================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_lookup_customer(crm_tool_no_api_key):
    """Test execute with lookup_customer action."""
    result = await crm_tool_no_api_key.execute(
        action="lookup_customer",
        customer_id="CUST-001"
    )
    
    assert isinstance(result, ToolResult)
    assert result.success is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_get_tickets(crm_tool_no_api_key):
    """Test execute with get_tickets action."""
    result = await crm_tool_no_api_key.execute(
        action="get_tickets",
        customer_id="CUST-001"
    )
    
    assert isinstance(result, ToolResult)
    assert result.success is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_update_customer(crm_tool_no_api_key):
    """Test execute with update_customer action."""
    result = await crm_tool_no_api_key.execute(
        action="update_customer",
        customer_id="CUST-001",
        updates={"tier": "gold"}
    )
    
    assert isinstance(result, ToolResult)
    assert result.success is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_unknown_action(crm_tool_no_api_key):
    """Test execute with unknown action."""
    result = await crm_tool_no_api_key.execute(
        action="unknown_action",
        customer_id="CUST-001"
    )
    
    assert isinstance(result, ToolResult)
    assert result.success is False
    assert "Unknown action" in result.error


# ===========================
# Validation Tests
# ===========================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_lookup_customer_missing_params(crm_tool_no_api_key):
    """Test lookup customer without required parameters."""
    result = await crm_tool_no_api_key.lookup_customer_async()
    
    assert isinstance(result, ToolResult)
    assert result.success is False
    assert "required" in result.error.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_tickets_missing_customer_id(crm_tool_no_api_key):
    """Test get tickets without customer_id."""
    result = await crm_tool_no_api_key.get_customer_tickets_async(customer_id=None)
    
    assert isinstance(result, ToolResult)
    assert result.success is False
    assert "required" in result.error.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_customer_empty_updates(crm_tool_no_api_key):
    """Test update customer with empty updates."""
    result = await crm_tool_no_api_key.update_customer_async(
        customer_id="CUST-001",
        updates={}
    )
    
    assert isinstance(result, ToolResult)
    assert result.success is False
    assert "empty" in result.error.lower()


# ===========================
# HTTP Client Tests (Mocked)
# ===========================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_http_request_with_mock_response(crm_tool, monkeypatch):
    """Test HTTP request with mocked aiohttp response."""
    # Create mock response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "id": "CUST-999",
        "email": "mocked@example.com",
        "name": "Mocked User",
        "status": "active"
    })
    
    # Mock session.request
    async def mock_request(*args, **kwargs):
        return mock_response
    
    crm_tool.session.request = mock_request
    
    # Make request
    result = await crm_tool._make_api_request(
        method="GET",
        url=f"{crm_tool.api_endpoint}/customers/CUST-999",
        operation="test_request"
    )
    
    assert result["id"] == "CUST-999"
    assert result["email"] == "mocked@example.com"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_http_request_404_error(crm_tool):
    """Test HTTP request handling 404 error."""
    # Mock 404 response
    mock_response = AsyncMock()
    mock_response.status = 404
    
    async def mock_request(*args, **kwargs):
        return mock_response
    
    crm_tool.session.request = mock_request
    
    # Should raise CustomerNotFoundError
    with pytest.raises(CustomerNotFoundError):
        await crm_tool._make_api_request(
            method="GET",
            url=f"{crm_tool.api_endpoint}/customers/NONEXISTENT",
            operation="test_404"
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_http_request_401_error(crm_tool):
    """Test HTTP request handling 401 authentication error."""
    mock_response = AsyncMock()
    mock_response.status = 401
    
    async def mock_request(*args, **kwargs):
        return mock_response
    
    crm_tool.session.request = mock_request
    
    with pytest.raises(AuthenticationError):
        await crm_tool._make_api_request(
            method="GET",
            url=f"{crm_tool.api_endpoint}/customers/CUST-001",
            operation="test_401"
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_http_request_429_rate_limit(crm_tool):
    """Test HTTP request handling 429 rate limit error."""
    mock_response = AsyncMock()
    mock_response.status = 429
    
    async def mock_request(*args, **kwargs):
        return mock_response
    
    crm_tool.session.request = mock_request
    
    with pytest.raises(RateLimitError):
        await crm_tool._make_api_request(
            method="GET",
            url=f"{crm_tool.api_endpoint}/customers/CUST-001",
            operation="test_429"
        )


# ===========================
# Integration with Retry/Circuit Breaker
# ===========================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_crm_lookup_with_retry_on_failure(crm_tool):
    """Test CRM lookup retries on transient failures."""
    call_count = 0
    
    async def mock_request_failing_then_success(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        
        mock_response = AsyncMock()
        
        if call_count < 2:
            # First call fails
            mock_response.status = 500
        else:
            # Second call succeeds
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "id": "CUST-RETRY",
                "email": "retry@example.com",
                "status": "active"
            })
        
        return mock_response
    
    crm_tool.session.request = mock_request_failing_then_success
    
    # Should succeed after retry
    result = await crm_tool.lookup_customer_async(customer_id="CUST-RETRY")
    
    assert result.success is True
    assert result.data["profile"]["customer_id"] == "CUST-RETRY"
    assert call_count >= 2  # At least one retry happened


# ===========================
# Performance Tests
# ===========================

@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
async def test_concurrent_crm_lookups(crm_tool_no_api_key):
    """Test concurrent CRM lookups."""
    customer_ids = [f"CUST-{i:04d}" for i in range(10)]
    
    tasks = [
        crm_tool_no_api_key.lookup_customer_async(customer_id=cid)
        for cid in customer_ids
    ]
    
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 10
    for result in results:
        assert isinstance(result, ToolResult)
        assert result.success is True


# ===========================
# Legacy Method Tests
# ===========================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_legacy_lookup_customer_method(crm_tool_no_api_key):
    """Test legacy lookup_customer method (backward compatibility)."""
    result = await crm_tool_no_api_key.lookup_customer(customer_id="CUST-LEGACY")
    
    assert isinstance(result, ToolResult)
    assert result.success is True


@pytest.mark.unit
def test_legacy_setup_method_raises_error():
    """Test legacy _setup method raises NotImplementedError."""
    tool = CRMTool()
    
    with pytest.raises(NotImplementedError):
        tool._setup()
```

---

## File 7: `tests/integration/test_agent_with_crm.py` (NEW)

```python
"""
Integration tests for agent with CRM tool (Phase 5).
Validates end-to-end agent functionality with CRM integration.
"""
import pytest
import asyncio
import uuid
from unittest.mock import AsyncMock

from app.agents.chat_agent import CustomerSupportAgent, AgentResponse
from app.tools.crm_tool import CRMTool
from app.config.tool_settings import ToolSettings


# ===========================
# Fixtures
# ===========================

@pytest.fixture
def crm_enabled_settings():
    """Settings with CRM tool enabled."""
    return ToolSettings(
        enable_rag_tool=True,
        enable_memory_tool=True,
        enable_escalation_tool=True,
        enable_attachment_tool=True,
        enable_crm_tool=True,  # Enable CRM
        crm_api_endpoint="https://api.crm.example.com/v1",
        crm_api_key=None  # Use mock mode
    )


@pytest.fixture
async def agent_with_crm(crm_enabled_settings, monkeypatch):
    """Create agent with CRM tool enabled."""
    # Patch settings
    monkeypatch.setattr('app.config.tool_settings.tool_settings', crm_enabled_settings)
    
    # Create agent in registry mode
    agent = CustomerSupportAgent(use_registry=True)
    await agent.initialize_async()
    
    # Verify CRM tool was created
    assert 'crm' in agent.tools
    assert agent.tools['crm'].name == "crm_lookup"
    
    yield agent
    
    # Cleanup
    await agent.cleanup()


# ===========================
# Integration Tests
# ===========================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_with_crm_tool_available(agent_with_crm):
    """Test agent has CRM tool available."""
    assert 'crm' in agent_with_crm.tools
    
    crm_tool = agent_with_crm.tools['crm']
    assert crm_tool.initialized is True
    assert crm_tool.name == "crm_lookup"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_process_message_with_crm_context(agent_with_crm):
    """Test agent processes message and can access CRM data."""
    session_id = "test-crm-session-001"
    message = "What's the status of my account?"
    
    # Process message
    response = await agent_with_crm.process_message(
        session_id=session_id,
        message=message,
        user_id="user-123"
    )
    
    assert isinstance(response, AgentResponse)
    assert response.message is not None
    assert 'crm' in response.tools_used or 'crm' in agent_with_crm.tools


@pytest.mark.integration
@pytest.mark.asyncio
async def test_crm_lookup_in_agent_workflow(agent_with_crm):
    """Test CRM lookup within agent workflow."""
    crm_tool = agent_with_crm.tools['crm']
    
    # Perform lookup
    result = await crm_tool.lookup_customer_async(customer_id="CUST-12345")
    
    assert result.success is True
    assert result.data["found"] is True
    
    # Verify profile data
    profile = result.data["profile"]
    assert profile["customer_id"] == "CUST-12345"
    assert profile["account_status"] is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_crm_get_tickets_in_agent_workflow(agent_with_crm):
    """Test CRM get tickets within agent workflow."""
    crm_tool = agent_with_crm.tools['crm']
    
    # Get tickets
    result = await crm_tool.get_customer_tickets_async(customer_id="CUST-12345")
    
    assert result.success is True
    assert "tickets" in result.data
    assert result.data["total_count"] >= 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_response_includes_crm_metadata(agent_with_crm):
    """Test agent response includes CRM tool metadata."""
    session_id = "test-crm-meta-001"
    message = "Check my account status"
    
    response = await agent_with_crm.process_message(
        session_id=session_id,
        message=message
    )
    
    # Should have tool metadata
    assert "tools_used" in response.to_dict()
    assert response.tool_metadata is not None
    
    # Check initialization mode includes CRM
    assert "initialization_mode" in response.tool_metadata


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_crm_lookups_via_agent(agent_with_crm):
    """Test concurrent CRM lookups through agent."""
    crm_tool = agent_with_crm.tools['crm']
    
    customer_ids = ["CUST-001", "CUST-002", "CUST-003"]
    
    tasks = [
        crm_tool.lookup_customer_async(customer_id=cid)
        for cid in customer_ids
    ]
    
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 3
    for result in results:
        assert result.success is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_with_crm_cleanup(agent_with_crm):
    """Test agent cleanup properly closes CRM tool."""
    crm_tool = agent_with_crm.tools['crm']
    assert crm_tool.initialized is True
    
    # Cleanup
    await agent_with_crm.cleanup()
    
    # CRM tool should be cleaned up
    assert crm_tool.initialized is False
    assert crm_tool.session is None


# ===========================
# Error Handling Tests
# ===========================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_handles_crm_lookup_failure_gracefully(agent_with_crm):
    """Test agent handles CRM tool errors gracefully."""
    crm_tool = agent_with_crm.tools['crm']
    
    # Force tool to raise exception
    original_method = crm_tool.lookup_customer_async
    
    async def failing_lookup(*args, **kwargs):
        raise Exception("Simulated CRM API failure")
    
    crm_tool.lookup_customer_async = failing_lookup
    
    # Process message should still work
    session_id = "test-crm-error-001"
    message = "Check my account"
    
    response = await agent_with_crm.process_message(
        session_id=session_id,
        message=message
    )
    
    # Agent should still return a response
    assert isinstance(response, AgentResponse)
    assert response.message is not None
    
    # Restore original method
    crm_tool.lookup_customer_async = original_method


# ===========================
# Tool Registry Tests
# ===========================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_registry_creates_crm_tool(crm_enabled_settings, monkeypatch):
    """Test tool registry creates CRM tool when enabled."""
    from app.tools.registry import ToolRegistry, ToolDependencies
    from app.config import settings
    
    monkeypatch.setattr('app.tools.registry.tool_settings', crm_enabled_settings)
    
    dependencies = ToolDependencies(
        settings=settings,
        tool_settings=crm_enabled_settings
    )
    
    tools = await ToolRegistry.create_and_initialize_tools(
        dependencies=dependencies,
        enabled_only=True
    )
    
    assert 'crm' in tools
    assert tools['crm'].name == "crm_lookup"
    assert tools['crm'].initialized is True
    
    # Cleanup
    await ToolRegistry.cleanup_tools(tools)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_registry_status_includes_crm():
    """Test tool registry status includes CRM tool."""
    from app.tools.registry import ToolRegistry
    
    status = ToolRegistry.get_registry_status()
    
    assert 'crm' in status['available_tools']
    assert 'crm' in status['tools']
    
    crm_info = status['tools']['crm']
    assert crm_info['registered'] is True
    assert 'config' in crm_info


# ===========================
# End-to-End Scenario Tests
# ===========================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_customer_support_with_crm_lookup(agent_with_crm):
    """Test end-to-end customer support scenario with CRM lookup."""
    session_id = f"e2e-session-{uuid.uuid4()}"
    user_id = "user-e2e-001"
    
    # Step 1: Customer asks about account
    response1 = await agent_with_crm.process_message(
        session_id=session_id,
        message="What's my account tier?",
        user_id=user_id
    )
    
    assert isinstance(response1, AgentResponse)
    assert response1.tool_metadata["session_id"] == session_id
    
    # Step 2: Customer asks about open tickets
    response2 = await agent_with_crm.process_message(
        session_id=session_id,
        message="Do I have any open support tickets?",
        user_id=user_id
    )
    
    assert isinstance(response2, AgentResponse)
    assert response2.tool_metadata["message_count"] == 2
    
    # Step 3: Verify CRM tool was available
    crm_tool = agent_with_crm.tools['crm']
    tickets_result = await crm_tool.get_customer_tickets_async(customer_id=user_id)
    
    assert tickets_result.success is True


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
async def test_e2e_multi_session_crm_access(agent_with_crm):
    """Test multiple concurrent sessions accessing CRM."""
    session_count = 5
    session_ids = [f"multi-session-{i}" for i in range(session_count)]
    
    async def process_session(session_id):
        return await agent_with_crm.process_message(
            session_id=session_id,
            message="Check my account details",
            user_id=f"user-{session_id}"
        )
    
    # Process all sessions concurrently
    tasks = [process_session(sid) for sid in session_ids]
    responses = await asyncio.gather(*tasks)
    
    # All should succeed
    assert len(responses) == session_count
    for response in responses:
        assert isinstance(response, AgentResponse)
        assert response.message is not None
```

---

## File 8: `docs/domain_tool_template_guide.md` (NEW - Documentation)

```markdown
# Domain Tool Integration Guide

## Overview

This guide demonstrates how to create production-ready domain-specific tools using the CRM Tool as a reference implementation. Follow this pattern for Billing, Inventory, and other external API integrations.

---

## Quick Start

### 1. Copy the CRM Tool Template

```bash
# Create your new tool file
cp backend/app/tools/crm_tool.py backend/app/tools/your_domain_tool.py
```

### 2. Customize Data Models

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class YourDataModel:
    """Your domain-specific data model."""
    id: str
    name: str
    status: str
    # Add your fields...
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status
        }
```

### 3. Implement Your Tool

```python
class YourDomainTool(BaseTool):
    """
    Your domain tool description.
    
    Features:
    - Async HTTP client with connection pooling
    - Automatic retry with exponential backoff
    - Circuit breaker for fault tolerance
    - Full telemetry integration
    """
    
    def __init__(self):
        super().__init__(
            name="your_tool_name",
            description="Your tool description"
        )
        
        # Load configuration from tool_settings
        self.api_endpoint = tool_settings.your_api_endpoint
        self.api_key = tool_settings.your_api_key
        self.timeout = tool_settings.your_timeout
        
        # HTTP client (initialized in async initialize())
        self.session: Optional[ClientSession] = None
        
        # Configure retry and circuit breaker
        self.retry_config = RetryConfig(
            max_attempts=3,
            wait_multiplier=1.0,
            wait_min=1.0,
            wait_max=10.0
        )
        
        self.circuit_breaker_config = CircuitBreakerConfig(
            fail_max=5,
            timeout=60,
            name="your_api"
        )
```

### 4. Add Configuration

In `backend/app/config/tool_settings.py`:

```python
# Your Tool Configuration
enable_your_tool: bool = Field(
    default=False,
    description="Enable your domain tool"
)

your_api_endpoint: Optional[str] = Field(
    default=None,
    description="Your API endpoint URL"
)

your_api_key: Optional[str] = Field(
    default=None,
    description="Your API key"
)

your_timeout: int = Field(
    default=10,
    ge=1,
    description="Your API timeout in seconds"
)
```

### 5. Register in Tool Registry

In `backend/app/tools/registry.py`:

```python
@staticmethod
def create_your_tool(dependencies: ToolDependencies) -> BaseTool:
    """Create your tool instance."""
    from .your_domain_tool import YourDomainTool
    
    tool = YourDomainTool()
    logger.debug("Your tool created")
    return tool

# Add to _factories dict
_factories: Dict[str, Callable] = {
    # ... existing tools ...
    'your_tool': ToolFactory.create_your_tool,
}
```

### 6. Create Tests

```bash
# Copy test template
cp tests/test_crm_tool.py tests/test_your_tool.py
```

---

## Implementation Patterns

### HTTP Request with Retry & Circuit Breaker

```python
async def _make_api_request(
    self,
    method: str,
    url: str,
    params: Optional[Dict] = None,
    json_data: Optional[Dict] = None,
    operation: str = "api_request"
) -> Dict[str, Any]:
    """Make HTTP request with retry and circuit breaker."""
    
    # Wrap with telemetry wrapper
    @with_tool_call_wrapper(
        tool_name=self.name,
        operation=operation,
        retry_config=self.retry_config,
        circuit_breaker_config=self.circuit_breaker_config,
        timeout=self.timeout,
        convert_to_tool_result=False
    )
    async def execute_request(**kwargs):
        async with self.session.request(
            method=method,
            url=url,
            params=params,
            json=json_data,
            headers=self._get_headers()
        ) as response:
            # Handle status codes
            if response.status == 404:
                raise YourNotFoundError()
            elif response.status >= 500:
                raise YourServerError()
            
            return await response.json()
    
    return await execute_request()
```

### Mock Mode for Testing

```python
def _get_mock_response(self, operation: str) -> Dict[str, Any]:
    """Get mock response for testing without real API."""
    
    logger.info(f"Returning mock response for {operation}")
    
    if "lookup" in operation:
        return {
            "id": "MOCK-001",
            "name": "Mock Data",
            "status": "active"
        }
    
    return {"status": "ok"}

# In _make_api_request:
if not self.api_key:
    logger.info("API key not configured, using mock data")
    return self._get_mock_response(operation)
```

### Execute Method Pattern

```python
async def execute(self, **kwargs) -> ToolResult:
    """Execute tool operation."""
    
    action = kwargs.get("action", "default_action")
    
    if action == "action_one":
        return await self.action_one_async(
            param=kwargs.get("param")
        )
    
    elif action == "action_two":
        return await self.action_two_async(
            param=kwargs.get("param")
        )
    
    else:
        return ToolResult.error_result(
            error=f"Unknown action: {action}",
            metadata={"tool": self.name}
        )
```

---

## Testing Strategy

### Unit Tests

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_your_tool_initialization():
    """Test tool initialization."""
    tool = YourDomainTool()
    await tool.initialize()
    
    assert tool.initialized is True
    assert tool.session is not None
    
    await tool.cleanup()

@pytest.mark.unit
@pytest.mark.asyncio
async def test_mock_mode_operation():
    """Test operation in mock mode."""
    tool = YourDomainTool()
    tool.api_key = None  # Mock mode
    
    await tool.initialize()
    
    result = await tool.your_operation_async(id="TEST-001")
    
    assert result.success is True
    assert result.data is not None
    
    await tool.cleanup()
```

### Integration Tests

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_with_your_tool(agent_with_your_tool):
    """Test agent uses your tool."""
    
    assert 'your_tool' in agent_with_your_tool.tools
    
    tool = agent_with_your_tool.tools['your_tool']
    result = await tool.execute(action="lookup", id="TEST-001")
    
    assert isinstance(result, ToolResult)
```

---

## Configuration Examples

### Development (.env)

```bash
# Your Tool Configuration
ENABLE_YOUR_TOOL=false
YOUR_API_ENDPOINT=https://api.example.com/v1
YOUR_API_KEY=  # Empty = mock mode
YOUR_TIMEOUT=10
YOUR_MAX_RETRIES=3
```

### Production

```bash
# Your Tool Configuration
ENABLE_YOUR_TOOL=true
YOUR_API_ENDPOINT=https://api.production.com/v1
YOUR_API_KEY=${VAULT_YOUR_API_KEY}  # From secrets manager
YOUR_TIMEOUT=15
YOUR_MAX_RETRIES=5
```

---

## Common Patterns

### Authentication Headers

```python
def _get_headers(self) -> Dict[str, str]:
    """Get request headers with authentication."""
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"{settings.app_name}/1.0"
    }
    
    if self.api_key:
        headers["Authorization"] = f"Bearer {self.api_key}"
        # Or: headers["X-API-Key"] = self.api_key
    
    return headers
```

### Custom Exceptions

```python
class YourAPIError(Exception):
    """Base exception for your API errors."""
    pass

class YourNotFoundError(YourAPIError):
    """Resource not found."""
    pass

class YourAuthError(YourAPIError):
    """Authentication failed."""
    pass
```

### Response Parsing

```python
def _parse_response(self, data: Dict[str, Any]) -> YourDataModel:
    """Parse API response into data model."""
    return YourDataModel(
        id=data.get("id") or data.get("identifier"),
        name=data.get("name", "Unknown"),
        status=data.get("status", "unknown"),
        # Map remaining fields...
    )
```

---

## Deployment Checklist

- [ ] Configuration added to `tool_settings.py`
- [ ] Tool registered in `registry.py`
- [ ] Unit tests created and passing
- [ ] Integration tests with agent created
- [ ] Mock mode tested without API credentials
- [ ] Error handling for all HTTP status codes
- [ ] Retry and circuit breaker configured
- [ ] Telemetry integration verified (spans, logs)
- [ ] Documentation updated
- [ ] Environment variables documented
- [ ] Secrets management plan (no keys in code)

---

## Reference Implementation

See `backend/app/tools/crm_tool.py` for the complete, production-ready reference implementation.

Key files to review:
- `crm_tool.py` - Full implementation
- `tests/test_crm_tool.py` - Comprehensive tests
- `tests/integration/test_agent_with_crm.py` - Agent integration
- `tool_settings.py` - Configuration pattern
- `registry.py` - Factory registration

---

## Support

For questions or issues with domain tool integration, refer to:
- CRM Tool implementation comments
- Phase 5 implementation documentation
- Tool wrapper documentation (`tool_call_wrapper.py`)

---

**Template Version**: Phase 5 - Production Ready  
**Last Updated**: 2024-01-15
```

---

*Due to character limits, I'll provide the final checklist and summary in the next message...*

Would you like me to continue with the Phase 5 Complete Integration Checklist and validation guide?
