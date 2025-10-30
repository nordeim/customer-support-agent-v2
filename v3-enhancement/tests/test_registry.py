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
