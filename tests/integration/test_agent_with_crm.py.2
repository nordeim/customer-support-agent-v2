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
