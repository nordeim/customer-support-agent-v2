"""
Integration tests for agent with telemetry-wrapped tool calls (Phase 3).
Validates end-to-end message processing with observability.
"""
import pytest
import asyncio
import uuid
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from app.agents.chat_agent import CustomerSupportAgent, AgentResponse
from app.tools.base_tool import ToolResult
from app.tools.tool_call_wrapper import (
    get_tool_metrics,
    reset_circuit_breaker,
    RetryConfig,
    CircuitBreakerConfig
)


# ===========================
# Fixtures
# ===========================

@pytest.fixture
def mock_tool_with_telemetry():
    """Create mock tool that returns ToolResult."""
    tool = MagicMock()
    tool.name = "mock_tool"
    tool.initialized = True
    
    async def mock_execute(**kwargs):
        return ToolResult.success_result(
            data={"result": "success"},
            metadata={"tool": "mock_tool"}
        )
    
    tool.execute = mock_execute
    return tool


@pytest.fixture
async def agent_with_mocked_tools():
    """Create agent with mocked tools for testing."""
    agent = CustomerSupportAgent(use_registry=False)
    
    # Mock RAG tool
    rag_tool = MagicMock()
    rag_tool.name = "rag_search"
    
    async def mock_search(**kwargs):
        return {
            "sources": [
                {
                    "content": "Mock RAG result",
                    "metadata": {"type": "test"},
                    "relevance_score": 0.95,
                    "rank": 1
                }
            ],
            "total_results": 1
        }
    
    rag_tool.search = mock_search
    agent.tools['rag'] = rag_tool
    
    # Mock Memory tool
    memory_tool = MagicMock()
    memory_tool.name = "memory_management"
    memory_tool.summarize_session = AsyncMock(return_value="Mock session summary")
    memory_tool.retrieve_memories = AsyncMock(return_value=[])
    memory_tool.store_memory = AsyncMock(return_value={"success": True})
    agent.tools['memory'] = memory_tool
    
    # Mock Escalation tool
    escalation_tool = MagicMock()
    escalation_tool.name = "escalation_check"
    
    async def mock_should_escalate(**kwargs):
        return {
            "escalate": False,
            "confidence": 0.3,
            "reasons": []
        }
    
    escalation_tool.should_escalate = mock_should_escalate
    agent.tools['escalation'] = escalation_tool
    
    yield agent
    
    await agent.cleanup()


# ===========================
# Tool Call Wrapper Tests
# ===========================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_call_with_success(mock_tool_with_telemetry):
    """Test successful tool call with wrapper."""
    from app.tools.tool_call_wrapper import call_tool_with_wrapper
    
    result = await call_tool_with_wrapper(
        tool=mock_tool_with_telemetry,
        method_name='execute',
        request_id='test-request-123',
        session_id='test-session-456'
    )
    
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.data["result"] == "success"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_call_with_retry():
    """Test tool call retry on failure."""
    from app.tools.tool_call_wrapper import call_tool_with_wrapper
    
    # Create tool that fails twice then succeeds
    call_count = 0
    
    tool = MagicMock()
    tool.name = "retry_test_tool"
    
    async def failing_method(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Temporary failure")
        return ToolResult.success_result(data={"attempts": call_count})
    
    tool.test_method = failing_method
    
    retry_config = RetryConfig(
        max_attempts=3,
        wait_min=0.1,
        wait_max=0.5
    )
    
    result = await call_tool_with_wrapper(
        tool=tool,
        method_name='test_method',
        retry_config=retry_config,
        timeout=5.0
    )
    
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.data["attempts"] == 3
    assert call_count == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_call_with_timeout():
    """Test tool call timeout handling."""
    from app.tools.tool_call_wrapper import call_tool_with_wrapper, ToolTimeoutError
    
    tool = MagicMock()
    tool.name = "timeout_test_tool"
    
    async def slow_method(**kwargs):
        await asyncio.sleep(2.0)
        return ToolResult.success_result(data={})
    
    tool.slow_method = slow_method
    
    result = await call_tool_with_wrapper(
        tool=tool,
        method_name='slow_method',
        timeout=0.5,  # Will timeout
        convert_to_tool_result=True
    )
    
    assert isinstance(result, ToolResult)
    assert result.success is False
    assert "timed out" in result.error.lower()


@pytest.mark.unit
def test_circuit_breaker_metrics():
    """Test circuit breaker metrics retrieval."""
    from app.tools.tool_call_wrapper import get_circuit_breaker, get_tool_metrics
    
    # Create circuit breaker
    cb = get_circuit_breaker('test_tool')
    
    # Get metrics
    metrics = get_tool_metrics('test_tool')
    
    assert metrics['tool'] == 'test_tool'
    assert 'state' in metrics
    assert 'fail_counter' in metrics


@pytest.mark.unit
def test_circuit_breaker_reset():
    """Test circuit breaker reset."""
    from app.tools.tool_call_wrapper import reset_circuit_breaker, get_circuit_breaker
    
    cb = get_circuit_breaker('reset_test_tool')
    
    # Reset should not raise
    reset_circuit_breaker('reset_test_tool')


# ===========================
# Agent Integration Tests
# ===========================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_process_message_with_telemetry(agent_with_mocked_tools):
    """Test agent message processing with telemetry."""
    session_id = "test-session-001"
    message = "How do I reset my password?"
    request_id = str(uuid.uuid4())
    
    response = await agent_with_mocked_tools.process_message(
        session_id=session_id,
        message=message,
        request_id=request_id
    )
    
    assert isinstance(response, AgentResponse)
    assert response.tool_metadata["request_id"] == request_id
    assert response.tool_metadata["session_id"] == session_id
    assert "circuit_breaker_status" in response.tool_metadata


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_process_message_with_sources(agent_with_mocked_tools):
    """Test agent returns sources from RAG."""
    session_id = "test-session-002"
    message = "What is your refund policy?"
    
    response = await agent_with_mocked_tools.process_message(
        session_id=session_id,
        message=message
    )
    
    assert isinstance(response, AgentResponse)
    assert len(response.sources) > 0
    assert response.sources[0]["content"] == "Mock RAG result"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_context_tracking(agent_with_mocked_tools):
    """Test agent tracks context with request_id."""
    session_id = "test-session-003"
    request_id_1 = str(uuid.uuid4())
    request_id_2 = str(uuid.uuid4())
    
    # First message
    response1 = await agent_with_mocked_tools.process_message(
        session_id=session_id,
        message="First message",
        request_id=request_id_1
    )
    
    assert response1.tool_metadata["message_count"] == 1
    assert response1.tool_metadata["request_id"] == request_id_1
    
    # Second message
    response2 = await agent_with_mocked_tools.process_message(
        session_id=session_id,
        message="Second message",
        request_id=request_id_2
    )
    
    assert response2.tool_metadata["message_count"] == 2
    assert response2.tool_metadata["request_id"] == request_id_2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_concurrent_sessions(agent_with_mocked_tools):
    """Test agent handles concurrent sessions."""
    session_ids = ["session-001", "session-002", "session-003"]
    messages = ["Message 1", "Message 2", "Message 3"]
    
    tasks = [
        agent_with_mocked_tools.process_message(sid, msg)
        for sid, msg in zip(session_ids, messages)
    ]
    
    responses = await asyncio.gather(*tasks)
    
    assert len(responses) == 3
    for response in responses:
        assert isinstance(response, AgentResponse)
        assert response.message is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_streaming_with_telemetry(agent_with_mocked_tools):
    """Test agent streaming response with telemetry."""
    session_id = "test-session-004"
    message = "Test streaming"
    request_id = str(uuid.uuid4())
    
    chunks = []
    async for chunk in agent_with_mocked_tools.stream_response(
        session_id=session_id,
        message=message,
        request_id=request_id
    ):
        chunks.append(chunk)
    
    # Verify streaming chunks
    assert len(chunks) > 0
    assert chunks[0]["type"] == "start"
    assert chunks[0]["request_id"] == request_id
    assert chunks[-1]["type"] == "complete"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_error_handling_with_telemetry(agent_with_mocked_tools):
    """Test agent error handling with telemetry."""
    # Make RAG tool raise exception
    async def failing_search(**kwargs):
        raise Exception("Simulated RAG failure")
    
    agent_with_mocked_tools.tools['rag'].search = failing_search
    
    session_id = "test-session-005"
    message = "This will fail"
    request_id = str(uuid.uuid4())
    
    response = await agent_with_mocked_tools.process_message(
        session_id=session_id,
        message=message,
        request_id=request_id
    )
    
    # Agent should still return a response
    assert isinstance(response, AgentResponse)
    # Response should not have sources due to RAG failure
    assert len(response.sources) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_tool_metrics_tracking(agent_with_mocked_tools):
    """Test that tool metrics are tracked."""
    session_id = "test-session-006"
    message = "Test message"
    
    response = await agent_with_mocked_tools.process_message(
        session_id=session_id,
        message=message
    )
    
    # Check that circuit breaker status is included
    assert "circuit_breaker_status" in response.tool_metadata
    cb_status = response.tool_metadata["circuit_breaker_status"]
    
    # Should be a dict with tool metrics
    assert isinstance(cb_status, dict)


# ===========================
# Telemetry Context Tests
# ===========================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_call_context_logging():
    """Test tool call context creates proper logs."""
    from app.tools.tool_call_wrapper import tool_call_context
    
    async with tool_call_context(
        tool_name='test_tool',
        operation='test_operation',
        request_id='req-123',
        session_id='sess-456',
        custom_field='custom_value'
    ) as span:
        # Set custom attribute
        if hasattr(span, 'set_attribute'):
            span.set_attribute('test.attr', 'value')
        
        # Simulate work
        await asyncio.sleep(0.1)
    
    # Context manager should complete without errors
    assert True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_call_context_error_handling():
    """Test tool call context handles errors."""
    from app.tools.tool_call_wrapper import tool_call_context
    
    with pytest.raises(ValueError):
        async with tool_call_context(
            tool_name='error_tool',
            operation='fail_operation',
            request_id='req-error'
        ):
            raise ValueError("Simulated error")


# ===========================
# Performance Tests
# ===========================

@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
async def test_agent_performance_under_load(agent_with_mocked_tools):
    """Test agent performance with multiple concurrent requests."""
    import time
    
    session_ids = [f"perf-session-{i}" for i in range(10)]
    messages = [f"Performance test message {i}" for i in range(10)]
    
    start_time = time.time()
    
    tasks = [
        agent_with_mocked_tools.process_message(sid, msg)
        for sid, msg in zip(session_ids, messages)
    ]
    
    responses = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    assert len(responses) == 10
    assert total_time < 5.0  # Should complete in under 5 seconds
    
    # All responses should be successful
    for response in responses:
        assert isinstance(response, AgentResponse)
        assert response.processing_time < 2.0  # Each should be under 2 seconds
