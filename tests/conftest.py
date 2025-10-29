"""
Pytest configuration and shared fixtures for testing.
Provides test database, cache, settings overrides, and tool mocks.
"""
import pytest
import os
import tempfile
import asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator, Dict, Any
from unittest.mock import MagicMock, AsyncMock

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# Set testing environment before importing app
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"  # Test DB
os.environ["ENVIRONMENT"] = "testing"
os.environ["DEBUG"] = "true"
os.environ["ENABLE_TELEMETRY"] = "false"

from app.database import Base
from app.config import Settings, get_settings


# ===========================
# Event Loop Fixtures
# ===========================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ===========================
# Settings Fixtures
# ===========================

@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """
    Create test settings instance.
    Override default settings for testing environment.
    """
    return Settings(
        environment="testing",
        debug=True,
        database_url="sqlite:///:memory:",
        redis_url="redis://localhost:6379/15",
        enable_telemetry=False,
        cache_enabled=False,  # Disable Redis cache in tests by default
        rate_limit_enabled=False,
        # Tool settings
        rag_enabled=True,
        memory_enabled=True,
        escalation_enabled=True,
        # Agent settings
        agent_model="gpt-4o-mini",
        agent_temperature=0.7,
        agent_max_tokens=2000,
        # Development settings
        dev_mock_ai=True,  # Use mock AI responses in tests
        dev_sample_data=False  # Don't load sample data in tests
    )


@pytest.fixture
def settings_override(test_settings: Settings, monkeypatch):
    """
    Override settings for individual tests.
    Usage: settings_override({"cache_enabled": True})
    """
    def _override(overrides: Dict[str, Any]) -> Settings:
        for key, value in overrides.items():
            monkeypatch.setattr(test_settings, key, value)
        return test_settings
    
    return _override


# ===========================
# Database Fixtures
# ===========================

@pytest.fixture(scope="session")
def test_db_engine():
    """
    Create in-memory SQLite engine for testing.
    Scope: session (reused across all tests in session).
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False  # Set to True for SQL debugging
    )
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    yield engine
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def test_db_session(test_db_engine) -> Generator[Session, None, None]:
    """
    Create a new database session for each test function.
    Automatically rolls back changes after test.
    """
    connection = test_db_engine.connect()
    transaction = connection.begin()
    
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=connection
    )
    
    session = SessionLocal()
    
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def test_db_with_data(test_db_session: Session):
    """
    Create database session pre-populated with test data.
    """
    from app.models.session import Session as SessionModel
    from app.models.memory import Memory
    from app.models.message import Message
    from datetime import datetime, timedelta
    
    # Create test session
    test_session = SessionModel(
        id="test-session-001",
        user_id="test-user-001",
        status="active",
        created_at=datetime.utcnow()
    )
    test_db_session.add(test_session)
    
    # Create test memories
    memories = [
        Memory(
            session_id="test-session-001",
            content="User prefers email communication",
            content_type="preference",
            importance=0.8,
            created_at=datetime.utcnow() - timedelta(hours=2)
        ),
        Memory(
            session_id="test-session-001",
            content="User's account tier is premium",
            content_type="fact",
            importance=0.9,
            created_at=datetime.utcnow() - timedelta(hours=1)
        )
    ]
    test_db_session.add_all(memories)
    
    # Create test messages
    messages = [
        Message(
            session_id="test-session-001",
            role="user",
            content="How do I reset my password?",
            created_at=datetime.utcnow() - timedelta(minutes=30)
        ),
        Message(
            session_id="test-session-001",
            role="assistant",
            content="To reset your password, click 'Forgot Password' on the login page.",
            created_at=datetime.utcnow() - timedelta(minutes=29)
        )
    ]
    test_db_session.add_all(messages)
    
    test_db_session.commit()
    
    return test_db_session


# ===========================
# Cache Fixtures
# ===========================

@pytest.fixture
def fake_cache():
    """
    Fake in-memory cache for testing (no Redis required).
    Implements CacheService interface.
    """
    class FakeCache:
        def __init__(self):
            self._store: Dict[str, Any] = {}
            self.enabled = True
        
        async def get(self, key: str) -> Any:
            return self._store.get(key)
        
        async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
            self._store[key] = value
            return True
        
        async def delete(self, key: str) -> bool:
            if key in self._store:
                del self._store[key]
                return True
            return False
        
        async def clear_pattern(self, pattern: str) -> int:
            keys_to_delete = [k for k in self._store.keys() if pattern.replace("*", "") in k]
            for key in keys_to_delete:
                del self._store[key]
            return len(keys_to_delete)
        
        async def ping(self) -> bool:
            return True
        
        async def close(self) -> None:
            self._store.clear()
    
    return FakeCache()


# ===========================
# Tool Mock Fixtures
# ===========================

@pytest.fixture
def mock_rag_tool():
    """Mock RAG tool with synchronous interface (current contract)."""
    tool = MagicMock()
    tool.name = "rag_search"
    tool.initialized = True
    
    # Mock search method (sync)
    def mock_search(query: str, k: int = 5, **kwargs):
        return {
            "query": query,
            "sources": [
                {
                    "content": f"Mock result for: {query}",
                    "metadata": {"type": "mock"},
                    "relevance_score": 0.95,
                    "rank": 1
                }
            ],
            "total_results": 1
        }
    
    tool.search = mock_search
    
    # Mock add_documents method
    def mock_add_documents(documents, **kwargs):
        return {
            "success": True,
            "documents_added": len(documents),
            "chunks_created": len(documents)
        }
    
    tool.add_documents = mock_add_documents
    
    return tool


@pytest.fixture
def mock_memory_tool():
    """Mock Memory tool."""
    tool = MagicMock()
    tool.name = "memory"
    tool.initialized = True
    
    # Mock methods
    tool.store_memory = AsyncMock(return_value={"success": True})
    tool.retrieve_memories = AsyncMock(return_value=[
        {"content": "User prefers email", "importance": 0.8}
    ])
    tool.summarize_session = AsyncMock(return_value="User has been active for 2 hours")
    
    return tool


@pytest.fixture
def mock_escalation_tool():
    """Mock Escalation tool."""
    tool = MagicMock()
    tool.name = "escalation"
    tool.initialized = True
    
    # Mock methods
    tool.should_escalate = MagicMock(return_value={
        "escalate": False,
        "confidence": 0.3,
        "reasons": []
    })
    tool.create_escalation_ticket = MagicMock(return_value={
        "ticket_id": "TICKET-12345",
        "status": "created"
    })
    
    return tool


@pytest.fixture
def mock_tools_dict(mock_rag_tool, mock_memory_tool, mock_escalation_tool):
    """Complete mock tools dictionary for agent."""
    return {
        "rag": mock_rag_tool,
        "memory": mock_memory_tool,
        "escalation": mock_escalation_tool
    }


# ===========================
# Agent Fixtures
# ===========================

@pytest.fixture
def mock_agent(mock_tools_dict):
    """
    Mock CustomerSupportAgent for testing without full initialization.
    """
    agent = MagicMock()
    agent.tools = mock_tools_dict
    agent.contexts = {}
    
    # Mock process_message to return a simple response
    async def mock_process_message(session_id: str, message: str, **kwargs):
        from app.models.schemas import AgentResponse
        return AgentResponse(
            session_id=session_id,
            message="Mock response",
            sources=[],
            escalated=False,
            confidence=0.9,
            tool_metadata={}
        )
    
    agent.process_message = mock_process_message
    agent.cleanup = AsyncMock()
    
    return agent


# ===========================
# Temporary Directory Fixtures
# ===========================

@pytest.fixture
def temp_dir():
    """Create temporary directory for file operations in tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_chroma_dir():
    """Create temporary ChromaDB directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ===========================
# Utility Fixtures
# ===========================

@pytest.fixture
def sample_documents():
    """Sample documents for RAG testing."""
    return [
        "To reset your password, click 'Forgot Password' on the login page.",
        "Our refund policy allows returns within 30 days of purchase.",
        "Customer support is available 24/7 via chat or email.",
        "Premium members get free shipping on all orders.",
        "To track your order, use the tracking number in your confirmation email."
    ]


@pytest.fixture
def sample_messages():
    """Sample conversation messages for testing."""
    return [
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "Click 'Forgot Password' on the login page."},
        {"role": "user", "content": "What is your refund policy?"},
        {"role": "assistant", "content": "We offer full refunds within 30 days."}
    ]


# ===========================
# Pytest Configuration
# ===========================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "requires_redis: marks tests requiring Redis connection"
    )
    config.addinivalue_line(
        "markers", "requires_openai: marks tests requiring OpenAI API key"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on their location."""
    for item in items:
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
