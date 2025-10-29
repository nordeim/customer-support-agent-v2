"""
Smoke test for agent process_message flow.
Validates end-to-end message processing with mocked tools.
"""
import pytest
import asyncio
from app.models.schemas import AgentResponse


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_process_message_basic(mock_agent):
    """Test basic agent message processing."""
    session_id = "test-session-001"
    message = "How do I reset my password?"
    
    response = await mock_agent.process_message(session_id, message)
    
    assert isinstance(response, AgentResponse)
    assert response.session_id == session_id
    assert response.message is not None
    assert isinstance(response.message, str)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_response_structure(mock_agent):
    """Validate AgentResponse structure."""
    session_id = "test-session-002"
    message = "Test message"
    
    response = await mock_agent.process_message(session_id, message)
    
    # Check required fields
    assert hasattr(response, 'session_id')
    assert hasattr(response, 'message')
    assert hasattr(response, 'sources')
    assert hasattr(response, 'escalated')
    assert hasattr(response, 'confidence')
    assert hasattr(response, 'tool_metadata')
    
    # Check types
    assert isinstance(response.sources, list)
    assert isinstance(response.escalated, bool)
    assert isinstance(response.tool_metadata, dict)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_uses_tools(mock_tools_dict):
    """Verify agent has access to tools."""
    assert 'rag' in mock_tools_dict
    assert 'memory' in mock_tools_dict
    assert 'escalation' in mock_tools_dict
    
    # Test RAG tool
    rag_result = mock_tools_dict['rag'].search("test query")
    assert rag_result is not None
    assert 'sources' in rag_result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_cleanup(mock_agent):
    """Test agent cleanup method."""
    await mock_agent.cleanup()
    # Should not raise any exceptions
    assert True


@pytest.mark.unit
def test_agent_contexts_dict(mock_agent):
    """Verify agent has contexts dictionary."""
    assert hasattr(mock_agent, 'contexts')
    assert isinstance(mock_agent.contexts, dict)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_message_processing(mock_agent):
    """Test processing multiple messages concurrently."""
    session_ids = ["session-001", "session-002", "session-003"]
    messages = ["Message 1", "Message 2", "Message 3"]
    
    tasks = [
        mock_agent.process_message(sid, msg)
        for sid, msg in zip(session_ids, messages)
    ]
    
    responses = await asyncio.gather(*tasks)
    
    assert len(responses) == 3
    for response in responses:
        assert isinstance(response, AgentResponse)
