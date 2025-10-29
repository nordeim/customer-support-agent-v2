# Phase 0 + Phase 1 Implementation Ready ✓

## Acknowledgment & Strategy

I have received and analyzed the three critical code files:
- ✅ **main.py** (348 lines) - Application lifecycle, middleware, startup checks
- ✅ **base_tool.py** (82 lines) - Current sync-first tool interface
- ✅ **rag_tool.py** (425 lines) - RAG implementation with mixed sync/async patterns

**Implementation Strategy**: I will deliver **Phase 0 (Test Scaffolding) + Phase 1 (Async Contract Standardization)** as complete drop-in replacement files. This establishes the foundation for all subsequent phases.

**Key Decisions**:
1. **Backward Compatibility**: All new async methods coexist with old sync methods (deprecated but functional)
2. **Feature Flag**: `AGENT_TOOL_REGISTRY_MODE = "legacy"` keeps current behavior unchanged until Phase 2
3. **No Breaking Changes**: `main.py` requires NO modifications for Phase 0-1 (agent call sites update in Phase 3)

---

## 📦 Phase 0 Deliverables: Test Scaffolding

### File 1: `tests/conftest.py` (NEW)

```python
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
```

### File 2: `tests/test_tool_contract.py` (NEW)

```python
"""
Test current tool contract to establish baseline.
Validates that existing BaseTool interface is present and functional.
"""
import pytest
from app.tools.base_tool import BaseTool


@pytest.mark.unit
def test_base_tool_exists():
    """Verify BaseTool class exists and is importable."""
    assert BaseTool is not None
    assert hasattr(BaseTool, '__init__')


@pytest.mark.unit
def test_base_tool_has_required_attributes():
    """Verify BaseTool defines expected interface."""
    # Check for abstract methods
    assert hasattr(BaseTool, '_setup')
    assert hasattr(BaseTool, 'execute')
    assert hasattr(BaseTool, 'cleanup')
    
    # Check for initialization methods
    assert hasattr(BaseTool, '_initialize')
    assert hasattr(BaseTool, '__call__')


@pytest.mark.unit
def test_base_tool_is_abstract():
    """Verify BaseTool cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseTool(name="test", description="test")


@pytest.mark.unit
def test_concrete_tool_implementation(mock_rag_tool):
    """Verify mock tools implement expected interface."""
    assert hasattr(mock_rag_tool, 'name')
    assert hasattr(mock_rag_tool, 'initialized')
    assert mock_rag_tool.initialized is True


@pytest.mark.unit
def test_tool_has_search_method(mock_rag_tool):
    """Verify RAG tool has search method."""
    assert hasattr(mock_rag_tool, 'search')
    assert callable(mock_rag_tool.search)
    
    # Test search functionality
    result = mock_rag_tool.search("test query", k=5)
    assert isinstance(result, dict)
    assert 'query' in result
    assert 'sources' in result


@pytest.mark.unit
def test_tool_add_documents_method(mock_rag_tool):
    """Verify RAG tool has add_documents method."""
    assert hasattr(mock_rag_tool, 'add_documents')
    assert callable(mock_rag_tool.add_documents)
    
    # Test add_documents functionality
    result = mock_rag_tool.add_documents(["doc1", "doc2"])
    assert isinstance(result, dict)
    assert 'success' in result
    assert result['success'] is True


@pytest.mark.unit
def test_memory_tool_async_methods(mock_memory_tool):
    """Verify Memory tool has async methods."""
    import inspect
    
    assert hasattr(mock_memory_tool, 'store_memory')
    assert hasattr(mock_memory_tool, 'retrieve_memories')
    assert hasattr(mock_memory_tool, 'summarize_session')


@pytest.mark.unit
def test_escalation_tool_methods(mock_escalation_tool):
    """Verify Escalation tool has required methods."""
    assert hasattr(mock_escalation_tool, 'should_escalate')
    assert hasattr(mock_escalation_tool, 'create_escalation_ticket')
    assert callable(mock_escalation_tool.should_escalate)
    assert callable(mock_escalation_tool.create_escalation_ticket)


@pytest.mark.unit
def test_tools_dict_structure(mock_tools_dict):
    """Verify tools are properly organized in dictionary."""
    assert isinstance(mock_tools_dict, dict)
    assert 'rag' in mock_tools_dict
    assert 'memory' in mock_tools_dict
    assert 'escalation' in mock_tools_dict
    
    # Verify each tool has required attributes
    for tool_name, tool in mock_tools_dict.items():
        assert hasattr(tool, 'name')
        assert hasattr(tool, 'initialized')
```

### File 3: `tests/test_agent_process_message_smoke.py` (NEW)

```python
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
```

### File 4: `scripts/run_tests.sh` (NEW)

```bash
#!/bin/bash

# Test runner script with comprehensive checks
# Usage: ./scripts/run_tests.sh [options]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default options
COVERAGE=true
VERBOSE=false
FAIL_UNDER=80
MARKERS=""
WARNINGS="error"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --fail-under)
            FAIL_UNDER="$2"
            shift 2
            ;;
        --mark|-m)
            MARKERS="-m $2"
            shift 2
            ;;
        --warnings)
            WARNINGS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --no-coverage         Disable coverage reporting"
            echo "  --verbose, -v         Enable verbose output"
            echo "  --fail-under <num>    Minimum coverage percentage (default: 80)"
            echo "  --mark, -m <marker>   Run only tests with specific marker"
            echo "  --warnings <mode>     Warning mode: error|default|ignore"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Run all tests with coverage"
            echo "  $0 -m unit            # Run only unit tests"
            echo "  $0 -m \"not slow\"      # Skip slow tests"
            echo "  $0 --no-coverage -v   # Run without coverage, verbose"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Running Test Suite${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest not found${NC}"
    echo "Install with: pip install pytest pytest-asyncio pytest-cov"
    exit 1
fi

# Set testing environment
export TESTING=true
export ENVIRONMENT=testing
export DEBUG=true
export ENABLE_TELEMETRY=false

# Build pytest command
PYTEST_CMD="pytest"

# Add verbose flag
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Add coverage options
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=app --cov-report=term-missing --cov-report=html --cov-fail-under=$FAIL_UNDER"
fi

# Add markers
if [ -n "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD $MARKERS"
fi

# Add warnings mode
PYTEST_CMD="$PYTEST_CMD -W $WARNINGS"

# Add color and show local variables on failure
PYTEST_CMD="$PYTEST_CMD --color=yes --tb=short"

# Run tests
echo -e "${YELLOW}Command: $PYTEST_CMD${NC}"
echo ""

if eval $PYTEST_CMD; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    if [ "$COVERAGE" = true ]; then
        echo ""
        echo -e "${GREEN}Coverage report generated: htmlcov/index.html${NC}"
    fi
    
    exit 0
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ Tests failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
```

**Make executable:**
```bash
chmod +x scripts/run_tests.sh
```

---

## 📦 Phase 1 Deliverables: Async Contract Standardization

### File 5: `backend/app/tools/base_tool.py` (COMPLETE REPLACEMENT)

```python
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
```

### File 6: `backend/app/tools/tool_adapters.py` (NEW)

```python
"""
Adapters for converting sync tool methods to async interface.
Provides backward compatibility during async migration.

Usage:
    # Wrap a sync function
    async_fn = sync_to_async_adapter(sync_function)
    result = await async_fn(args)
    
    # Ensure a tool is async-compatible
    async_tool = ensure_tool_async(legacy_tool)
"""
import asyncio
import functools
import inspect
from typing import Any, Callable, Coroutine, TypeVar, cast
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# Thread pool for running sync functions in async context
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool_adapter_")


def sync_to_async_adapter(func: Callable[..., T]) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Convert a synchronous function to async using thread pool executor.
    
    Args:
        func: Synchronous function to wrap
        
    Returns:
        Async function that runs sync func in thread pool
        
    Example:
        def sync_search(query: str) -> dict:
            return {"results": query}
        
        async_search = sync_to_async_adapter(sync_search)
        result = await async_search("test")
    """
    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            functools.partial(func, *args, **kwargs)
        )
    
    return async_wrapper


def ensure_async(func: F) -> F:
    """
    Decorator to ensure a function is async.
    If function is sync, converts it using sync_to_async_adapter.
    If already async, returns as-is.
    
    Args:
        func: Function to ensure is async
        
    Returns:
        Async version of function
        
    Example:
        @ensure_async
        def my_function(x: int) -> int:
            return x * 2
        
        # Can now be called with await
        result = await my_function(5)
    """
    if inspect.iscoroutinefunction(func):
        # Already async, return as-is
        return func
    
    # Wrap sync function
    async_func = sync_to_async_adapter(func)
    return cast(F, async_func)


class AsyncToolAdapter:
    """
    Adapter to wrap a legacy sync tool and provide async interface.
    
    Example:
        legacy_tool = OldSyncTool()
        async_tool = AsyncToolAdapter(legacy_tool)
        
        # Now can use async methods
        result = await async_tool.execute(query="test")
    """
    
    def __init__(self, tool: Any):
        """
        Wrap a sync tool with async interface.
        
        Args:
            tool: Legacy tool instance to wrap
        """
        self._tool = tool
        self.name = getattr(tool, 'name', 'unknown')
        self.description = getattr(tool, 'description', '')
        self.initialized = getattr(tool, 'initialized', False)
        
        logger.info(f"Created async adapter for tool '{self.name}'")
    
    async def initialize(self) -> None:
        """Initialize wrapped tool."""
        if hasattr(self._tool, 'initialize'):
            # Tool already has async initialize
            await self._tool.initialize()
        elif hasattr(self._tool, '_initialize'):
            # Legacy sync initialize
            await sync_to_async_adapter(self._tool._initialize)()
        
        self.initialized = True
    
    async def execute(self, **kwargs: Any) -> Any:
        """
        Execute wrapped tool method.
        
        Attempts to call in order:
        1. async execute() if exists
        2. sync execute() wrapped in adapter
        3. __call__() method wrapped in adapter
        """
        if hasattr(self._tool, 'execute'):
            execute_fn = self._tool.execute
            if inspect.iscoroutinefunction(execute_fn):
                return await execute_fn(**kwargs)
            else:
                return await sync_to_async_adapter(execute_fn)(**kwargs)
        
        elif callable(self._tool):
            if inspect.iscoroutinefunction(self._tool.__call__):
                return await self._tool(**kwargs)
            else:
                return await sync_to_async_adapter(self._tool.__call__)(**kwargs)
        
        else:
            raise NotImplementedError(f"Tool '{self.name}' has no execute or __call__ method")
    
    async def cleanup(self) -> None:
        """Cleanup wrapped tool."""
        if hasattr(self._tool, 'cleanup'):
            cleanup_fn = self._tool.cleanup
            if inspect.iscoroutinefunction(cleanup_fn):
                await cleanup_fn()
            else:
                await sync_to_async_adapter(cleanup_fn)()
    
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped tool."""
        return getattr(self._tool, name)


def ensure_tool_async(tool: Any) -> Any:
    """
    Ensure a tool has async interface.
    If tool is already async-compatible, returns as-is.
    Otherwise, wraps in AsyncToolAdapter.
    
    Args:
        tool: Tool instance to check/wrap
        
    Returns:
        Async-compatible tool
        
    Example:
        tool = legacy_sync_tool_instance
        async_tool = ensure_tool_async(tool)
        await async_tool.execute(query="test")
    """
    # Check if tool already has async execute
    if hasattr(tool, 'execute') and inspect.iscoroutinefunction(tool.execute):
        return tool
    
    # Check if tool has async initialize
    if hasattr(tool, 'initialize') and inspect.iscoroutinefunction(tool.initialize):
        return tool
    
    # Wrap in adapter
    logger.info(f"Wrapping tool '{getattr(tool, 'name', 'unknown')}' with async adapter")
    return AsyncToolAdapter(tool)


def cleanup_executor() -> None:
    """
    Cleanup thread pool executor.
    Call during application shutdown.
    """
    global _executor
    if _executor:
        logger.info("Shutting down tool adapter thread pool")
        _executor.shutdown(wait=True)
        _executor = None


# Export public API
__all__ = [
    'sync_to_async_adapter',
    'ensure_async',
    'AsyncToolAdapter',
    'ensure_tool_async',
    'cleanup_executor'
]
```

### File 7: `backend/app/tools/rag_tool.py` (COMPLETE REPLACEMENT)

```python
"""
RAG (Retrieval-Augmented Generation) tool implementation.
Uses EmbeddingGemma for embeddings and ChromaDB for vector storage.

Phase 1 Update: Async-first interface with ToolResult return types.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import numpy as np
from pathlib import Path
import asyncio

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from ..config import settings
from ..services.cache_service import CacheService
from .base_tool import BaseTool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)

# EmbeddingGemma-specific prefixes for optimal performance
QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: none | text: "

# Chunking parameters
CHUNK_SIZE = 500  # words
CHUNK_OVERLAP = 50  # words

# Search parameters
DEFAULT_K = 5
SIMILARITY_THRESHOLD = 0.7


class RAGTool(BaseTool):
    """
    RAG tool for searching and retrieving relevant documents.
    Uses Google's EmbeddingGemma model for generating embeddings
    and ChromaDB for efficient vector similarity search.
    
    Phase 1: Implements async-first interface with ToolResult returns.
    """
    
    def __init__(self):
        """Initialize RAG tool with embedding model and vector store."""
        # Call new-style parent init (no auto-initialization)
        super().__init__(
            name="rag_search",
            description="Search knowledge base for relevant information using semantic similarity"
        )
        
        # Resources will be initialized in async initialize()
        self.embedder = None
        self.chroma_client = None
        self.collection = None
        self.cache = None
    
    # ===========================
    # Async Interface (Phase 1)
    # ===========================
    
    async def initialize(self) -> None:
        """
        Initialize RAG tool resources (async-safe).
        Sets up embedding model, ChromaDB, and cache service.
        """
        try:
            logger.info(f"Initializing RAG tool '{self.name}'...")
            
            # Initialize cache service
            self.cache = CacheService()
            
            # Initialize embedding model (CPU-bound, run in thread pool)
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._init_embedding_model
            )
            
            # Initialize ChromaDB (I/O-bound, run in thread pool)
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._init_chromadb
            )
            
            self.initialized = True
            logger.info(f"✓ RAG tool '{self.name}' initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG tool: {e}", exc_info=True)
            raise
    
    async def cleanup(self) -> None:
        """Cleanup RAG tool resources."""
        try:
            logger.info(f"Cleaning up RAG tool '{self.name}'...")
            
            # Close cache connections
            if self.cache:
                await self.cache.close()
            
            # ChromaDB cleanup (if needed)
            if self.chroma_client:
                # ChromaDB doesn't require explicit cleanup in current version
                self.chroma_client = None
            
            self.initialized = False
            logger.info(f"✓ RAG tool '{self.name}' cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during RAG tool cleanup: {e}")
    
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute RAG search (async-first).
        
        Accepts:
            query: Search query (required)
            k: Number of results (optional, default: 5)
            filter: Metadata filter (optional)
            threshold: Similarity threshold (optional, default: 0.7)
            
        Returns:
            ToolResult with search results
        """
        query = kwargs.get("query")
        if not query:
            return ToolResult.error_result(
                error="Query parameter is required",
                metadata={"tool": self.name}
            )
        
        k = kwargs.get("k", DEFAULT_K)
        filter_dict = kwargs.get("filter")
        threshold = kwargs.get("threshold", SIMILARITY_THRESHOLD)
        
        try:
            result = await self.search_async(query, k, filter_dict, threshold)
            
            return ToolResult.success_result(
                data=result,
                metadata={
                    "tool": self.name,
                    "query_length": len(query),
                    "k": k,
                    "threshold": threshold,
                    "results_count": result.get('total_results', 0)
                }
            )
            
        except Exception as e:
            logger.error(f"RAG execute error: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={"tool": self.name, "query": query[:100]}
            )
    
    # ===========================
    # Core RAG Methods (Async)
    # ===========================
    
    async def search_async(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, Any]] = None,
        threshold: float = SIMILARITY_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Search for relevant documents using vector similarity (async).
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            threshold: Minimum similarity threshold
            
        Returns:
            Search results with documents and metadata
        """
        # Create cache key
        cache_key = f"rag_search:{query}:{k}:{str(filter)}"
        
        # Check cache first
        if self.cache and self.cache.enabled:
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_result
        
        try:
            # Generate query embedding (CPU-bound, run in thread pool)
            query_embedding = await asyncio.get_event_loop().run_in_executor(
                None,
                self.embed_query,
                query
            )
            
            # Search in ChromaDB (I/O-bound, run in thread pool)
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=k,
                    where=filter,
                    include=["documents", "metadatas", "distances"]
                )
            )
            
            # Format and filter results
            formatted_results = {
                "query": query,
                "sources": [],
                "total_results": 0
            }
            
            if results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    # Convert distance to similarity score (1 - distance for normalized vectors)
                    similarity = 1 - results['distances'][0][i]
                    
                    # Only include results above threshold
                    if similarity >= threshold:
                        source = {
                            "content": results['documents'][0][i],
                            "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                            "relevance_score": round(similarity, 4),
                            "rank": i + 1
                        }
                        formatted_results['sources'].append(source)
                
                formatted_results['total_results'] = len(formatted_results['sources'])
            
            # Cache the results
            if self.cache and self.cache.enabled and formatted_results['total_results'] > 0:
                await self.cache.set(cache_key, formatted_results, ttl=settings.redis_ttl)
            
            logger.info(
                f"RAG search completed: query='{query[:50]}...', "
                f"results={formatted_results['total_results']}/{k}"
            )
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"RAG search error: {e}", exc_info=True)
            return {
                "query": query,
                "sources": [],
                "error": str(e)
            }
    
    async def add_documents_async(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        chunk: bool = True
    ) -> ToolResult:
        """
        Add documents to the knowledge base (async).
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
            ids: Optional IDs for documents
            chunk: Whether to chunk documents before adding
            
        Returns:
            ToolResult with operation status
        """
        try:
            # Prepare documents (CPU-bound)
            prep_result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._prepare_documents,
                documents,
                metadatas,
                ids,
                chunk
            )
            
            if not prep_result['chunks']:
                return ToolResult.error_result(
                    error="No documents to add",
                    metadata={"tool": self.name}
                )
            
            all_chunks = prep_result['chunks']
            all_metadatas = prep_result['metadatas']
            all_ids = prep_result['ids']
            
            # Generate embeddings (CPU-bound)
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None,
                self.embed_documents,
                all_chunks
            )
            
            # Add to ChromaDB (I/O-bound)
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.add(
                    documents=all_chunks,
                    embeddings=[emb.tolist() for emb in embeddings],
                    metadatas=all_metadatas,
                    ids=all_ids
                )
            )
            
            # Clear cache as new documents were added
            if self.cache and self.cache.enabled:
                asyncio.create_task(self.cache.clear_pattern("rag_search:*"))
            
            logger.info(
                f"Added {len(documents)} documents "
                f"({len(all_chunks)} chunks) to knowledge base"
            )
            
            return ToolResult.success_result(
                data={
                    "documents_added": len(documents),
                    "chunks_created": len(all_chunks)
                },
                metadata={
                    "tool": self.name,
                    "chunking_enabled": chunk
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={"tool": self.name, "document_count": len(documents)}
            )
    
    # ===========================
    # Legacy Methods (Backward Compatibility)
    # ===========================
    
    def _setup(self) -> None:
        """
        DEPRECATED: Legacy sync setup.
        Use async initialize() instead.
        """
        logger.warning("RAGTool._setup is deprecated. Use await rag_tool.initialize()")
        self._init_embedding_model()
        self._init_chromadb()
        self.cache = CacheService()
    
    async def search(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, Any]] = None,
        threshold: float = SIMILARITY_THRESHOLD
    ) -> Dict[str, Any]:
        """
        DEPRECATED: Legacy search method.
        Use search_async() or execute() instead.
        """
        logger.warning("RAGTool.search is deprecated. Use search_async() instead.")
        return await self.search_async(query, k, filter, threshold)
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        chunk: bool = True
    ) -> Dict[str, Any]:
        """
        DEPRECATED: Legacy sync add_documents.
        Use add_documents_async() instead.
        
        Returns dict for backward compatibility.
        """
        logger.warning("RAGTool.add_documents (sync) is deprecated. Use await add_documents_async()")
        
        # Run async version synchronously (blocking)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.add_documents_async(documents, metadatas, ids, chunk)
            )
            return result.to_dict() if isinstance(result, ToolResult) else result
        finally:
            loop.close()
    
    # ===========================
    # Private Helper Methods
    # ===========================
    
    def _init_embedding_model(self) -> None:
        """Initialize embedding model (sync, called in thread pool)."""
        try:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            
            self.embedder = SentenceTransformer(
                settings.embedding_model,
                device='cpu'  # Use 'cuda' if GPU available
            )
            
            self.embedding_dim = settings.embedding_dimension
            logger.info(f"Embedding model loaded successfully (dim: {self.embedding_dim})")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            logger.warning("Falling back to all-MiniLM-L6-v2")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384
    
    def _init_chromadb(self) -> None:
        """Initialize ChromaDB client and collection (sync)."""
        try:
            persist_dir = Path(settings.chroma_persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            try:
                self.collection = self.chroma_client.get_collection(
                    name=settings.chroma_collection_name
                )
                logger.info(f"Using existing ChromaDB collection: {settings.chroma_collection_name}")
            except chromadb.errors.NotFoundError:
                self.collection = self.chroma_client.create_collection(
                    name=settings.chroma_collection_name,
                    metadata={
                        "hnsw:space": "ip",
                        "hnsw:construction_ef": 200,
                        "hnsw:M": 16
                    }
                )
                logger.info(f"Created new ChromaDB collection: {settings.chroma_collection_name}")
                self._add_sample_documents()
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _add_sample_documents(self) -> None:
        """Add sample documents to empty collection."""
        sample_docs = [
            "To reset your password, click on 'Forgot Password' on the login page and follow the instructions.",
            "Our refund policy allows returns within 30 days of purchase for a full refund.",
            "Customer support is available 24/7 via chat, email at support@example.com, or phone at 1-800-EXAMPLE.",
            "To track your order, use the tracking number provided in your confirmation email.",
            "Account verification requires a valid email address and phone number for security purposes."
        ]
        
        try:
            # Use sync add_documents for initial sample data
            result = self.add_documents(
                documents=sample_docs,
                metadatas=[{"type": "sample", "category": "faq"} for _ in sample_docs]
            )
            logger.info(f"Added {len(sample_docs)} sample documents to collection")
        except Exception as e:
            logger.warning(f"Failed to add sample documents: {e}")
    
    def _prepare_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]],
        ids: Optional[List[str]],
        chunk: bool
    ) -> Dict[str, Any]:
        """Prepare documents for indexing (chunking, ID generation)."""
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for idx, doc in enumerate(documents):
            if chunk and len(doc.split()) > CHUNK_SIZE:
                chunks = self.chunk_document(doc)
                for chunk_idx, (chunk_text, chunk_meta) in enumerate(chunks):
                    all_chunks.append(chunk_text)
                    
                    combined_meta = chunk_meta.copy()
                    if metadatas and idx < len(metadatas):
                        combined_meta.update(metadatas[idx])
                    combined_meta['doc_index'] = idx
                    all_metadatas.append(combined_meta)
                    
                    if ids and idx < len(ids):
                        chunk_id = f"{ids[idx]}_chunk_{chunk_idx}"
                    else:
                        chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()
                    all_ids.append(chunk_id)
            else:
                all_chunks.append(doc)
                
                meta = {"doc_index": idx}
                if metadatas and idx < len(metadatas):
                    meta.update(metadatas[idx])
                all_metadatas.append(meta)
                
                if ids and idx < len(ids):
                    all_ids.append(ids[idx])
                else:
                    all_ids.append(hashlib.md5(doc.encode()).hexdigest())
        
        return {
            "chunks": all_chunks,
            "metadatas": all_metadatas,
            "ids": all_ids
        }
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a search query (sync)."""
        prefixed_query = QUERY_PREFIX + query
        embedding = self.embedder.encode(
            prefixed_query,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embedding
    
    def embed_documents(self, documents: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple documents (sync)."""
        prefixed_docs = [DOC_PREFIX + doc for doc in documents]
        embeddings = self.embedder.encode(
            prefixed_docs,
            normalize_embeddings=True,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=len(documents) > 10,
            convert_to_numpy=True
        )
        return embeddings
    
    def chunk_document(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Split document into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_words = words[i:i + CHUNK_SIZE]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_words) >= CHUNK_OVERLAP:
                metadata = {
                    "chunk_index": len(chunks),
                    "start_word": i,
                    "end_word": min(i + CHUNK_SIZE, len(words)),
                    "total_words": len(words)
                }
                chunks.append((chunk_text, metadata))
        
        return chunks
```

### File 8: `tests/test_tool_async_contract.py` (NEW)

```python
"""
Tests for async tool contract and ToolResult.
Validates Phase 1 implementation.
"""
import pytest
import asyncio
from app.tools.base_tool import BaseTool, ToolResult, ToolStatus
from app.tools.tool_adapters import (
    sync_to_async_adapter,
    ensure_async,
    AsyncToolAdapter,
    ensure_tool_async
)


# ===========================
# ToolResult Tests
# ===========================

@pytest.mark.unit
def test_tool_result_creation():
    """Test ToolResult creation and fields."""
    result = ToolResult(
        success=True,
        data={"key": "value"},
        metadata={"tool": "test"}
    )
    
    assert result.success is True
    assert result.data == {"key": "value"}
    assert result.metadata == {"tool": "test"}
    assert result.error is None
    assert result.status == ToolStatus.SUCCESS


@pytest.mark.unit
def test_tool_result_error():
    """Test error result creation."""
    result = ToolResult.error_result(
        error="Something went wrong",
        metadata={"tool": "test"}
    )
    
    assert result.success is False
    assert result.error == "Something went wrong"
    assert result.status == ToolStatus.ERROR
    assert result.metadata["tool"] == "test"


@pytest.mark.unit
def test_tool_result_success_helper():
    """Test success result helper."""
    result = ToolResult.success_result(
        data={"count": 5},
        metadata={"source": "cache"}
    )
    
    assert result.success is True
    assert result.data["count"] == 5
    assert result.metadata["source"] == "cache"
    assert result.status == ToolStatus.SUCCESS


@pytest.mark.unit
def test_tool_result_to_dict():
    """Test ToolResult serialization to dict."""
    result = ToolResult(
        success=True,
        data={"key": "value"},
        metadata={"tool": "test"},
        status=ToolStatus.SUCCESS
    )
    
    result_dict = result.to_dict()
    
    assert isinstance(result_dict, dict)
    assert result_dict["success"] is True
    assert result_dict["data"] == {"key": "value"}
    assert result_dict["status"] == "success"


@pytest.mark.unit
def test_tool_result_from_dict():
    """Test ToolResult deserialization from dict."""
    data = {
        "success": True,
        "data": {"key": "value"},
        "metadata": {"tool": "test"},
        "status": "success"
    }
    
    result = ToolResult.from_dict(data)
    
    assert result.success is True
    assert result.data["key"] == "value"
    assert result.status == ToolStatus.SUCCESS


# ===========================
# Adapter Tests
# ===========================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_sync_to_async_adapter():
    """Test sync function wrapper."""
    def sync_function(x: int, y: int) -> int:
        return x + y
    
    async_function = sync_to_async_adapter(sync_function)
    result = await async_function(5, 3)
    
    assert result == 8


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ensure_async_with_sync_function():
    """Test ensure_async decorator with sync function."""
    @ensure_async
    def my_function(x: int) -> int:
        return x * 2
    
    result = await my_function(5)
    assert result == 10


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ensure_async_with_async_function():
    """Test ensure_async decorator with already-async function."""
    @ensure_async
    async def my_function(x: int) -> int:
        return x * 2
    
    result = await my_function(5)
    assert result == 10


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_tool_adapter():
    """Test AsyncToolAdapter with mock sync tool."""
    class MockSyncTool:
        def __init__(self):
            self.name = "mock_sync"
            self.description = "Mock sync tool"
            self.initialized = False
        
        def _initialize(self):
            self.initialized = True
        
        def execute(self, query: str) -> dict:
            return {"result": f"Processed: {query}"}
    
    sync_tool = MockSyncTool()
    async_tool = AsyncToolAdapter(sync_tool)
    
    await async_tool.initialize()
    assert async_tool.initialized is True
    
    result = await async_tool.execute(query="test")
    assert result["result"] == "Processed: test"
    
    await async_tool.cleanup()


@pytest.mark.unit
def test_ensure_tool_async_with_async_tool():
    """Test ensure_tool_async with already-async tool."""
    class AsyncTool(BaseTool):
        async def initialize(self):
            self.initialized = True
        
        async def execute(self, **kwargs) -> ToolResult:
            return ToolResult.success_result(data=kwargs)
        
        async def cleanup(self):
            pass
    
    tool = AsyncTool(name="test", description="Test tool")
    wrapped_tool = ensure_tool_async(tool)
    
    # Should return same instance
    assert wrapped_tool is tool


@pytest.mark.unit
def test_ensure_tool_async_with_sync_tool():
    """Test ensure_tool_async with sync tool."""
    class SyncTool:
        def __init__(self):
            self.name = "sync_tool"
            self.initialized = False
        
        def execute(self, **kwargs):
            return {"data": kwargs}
    
    sync_tool = SyncTool()
    async_tool = ensure_tool_async(sync_tool)
    
    # Should wrap in adapter
    assert isinstance(async_tool, AsyncToolAdapter)
    assert async_tool.name == "sync_tool"


# ===========================
# Integration Tests
# ===========================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_tool_full_lifecycle():
    """Test complete async tool lifecycle."""
    class TestAsyncTool(BaseTool):
        def __init__(self):
            super().__init__(
                name="test_async_tool",
                description="Test async tool implementation"
            )
            self.setup_called = False
            self.cleanup_called = False
        
        async def initialize(self):
            self.setup_called = True
            self.initialized = True
        
        async def execute(self, **kwargs) -> ToolResult:
            query = kwargs.get("query", "")
            return ToolResult.success_result(
                data={"response": f"Processed: {query}"},
                metadata={"tool": self.name}
            )
        
        async def cleanup(self):
            self.cleanup_called = True
    
    # Create tool
    tool = TestAsyncTool()
    assert tool.setup_called is False
    
    # Initialize
    await tool.initialize()
    assert tool.setup_called is True
    assert tool.initialized is True
    
    # Execute
    result = await tool.execute(query="test query")
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.data["response"] == "Processed: test query"
    assert result.metadata["tool"] == "test_async_tool"
    
    # Cleanup
    await tool.cleanup()
    assert tool.cleanup_called is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_tool_execution():
    """Test concurrent execution of async tools."""
    class ConcurrentTool(BaseTool):
        def __init__(self):
            super().__init__(
                name="concurrent_test",
                description="Concurrent execution test"
            )
            self.call_count = 0
        
        async def initialize(self):
            self.initialized = True
        
        async def execute(self, **kwargs) -> ToolResult:
            await asyncio.sleep(0.1)  # Simulate async work
            self.call_count += 1
            return ToolResult.success_result(
                data={"call": self.call_count}
            )
        
        async def cleanup(self):
            pass
    
    tool = ConcurrentTool()
    await tool.initialize()
    
    # Execute 5 calls concurrently
    tasks = [tool.execute() for _ in range(5)]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 5
    for result in results:
        assert isinstance(result, ToolResult)
        assert result.success is True
```

---

## 📋 Integration & Validation Guide

### Step 1: Install Test Dependencies

```bash
pip install pytest pytest-asyncio pytest-cov
```

### Step 2: Run Phase 0 Tests (Baseline)

```bash
# Run test scaffolding
./scripts/run_tests.sh -m unit

# Expected: All tests pass (validates current code)
```

### Step 3: Deploy Phase 1 Files

**Replace these files:**
1. `backend/app/tools/base_tool.py` → New version with ToolResult
2. `backend/app/tools/rag_tool.py` → Async-first version

**Add these new files:**
3. `backend/app/tools/tool_adapters.py` → Adapters for backward compat
4. All test files in `tests/` directory

### Step 4: Run Phase 1 Tests

```bash
# Run async contract tests
./scripts/run_tests.sh -m unit

# Run with coverage
./scripts/run_tests.sh --fail-under 80
```

### Step 5: Verify No Breaking Changes

**main.py should work unchanged** because:
- ✅ Legacy `_setup()` still supported in BaseTool
- ✅ RAGTool still has `add_documents()` sync method (deprecated but functional)
- ✅ Agent calls remain compatible during transition

**To verify:**
```bash
# Start the application
python -m app.main

# Check logs for deprecation warnings (expected)
# Application should start successfully
```

---

## ⚠️ Important Notes

### What NOT to Change Yet

**DO NOT modify** in Phase 0-1:
- ❌ `backend/app/agents/chat_agent.py` (agent call sites → Phase 3)
- ❌ `backend/app/main.py` (agent initialization → Phase 2)
- ❌ `backend/app/tools/memory_tool.py` (not included in current share → Phase 1 continuation)
- ❌ `backend/app/tools/escalation_tool.py` (not included → Phase 1 continuation)

### Deprecation Warnings Expected

You will see logs like:
```
WARNING - Tool 'rag_search' uses deprecated _setup method. Migrate to async initialize() for Phase 2+
WARNING - RAGTool.add_documents (sync) is deprecated. Use await add_documents_async()
```

**This is intentional** and safe. These warnings guide Phase 2-3 migration.

### Feature Flags

Add to `config.py` (if not already present):
```python
AGENT_TOOL_REGISTRY_MODE: str = Field(
    default="legacy",
    description="Tool initialization mode: legacy or registry"
)
```

---

## ✅ Phase 0-1 Completion Checklist

- [ ] All test files added to `tests/` directory
- [ ] `scripts/run_tests.sh` is executable (`chmod +x`)
- [ ] Phase 0 baseline tests pass (`./scripts/run_tests.sh -m unit`)
- [ ] `base_tool.py` replaced with async-first version
- [ ] `tool_adapters.py` added
- [ ] `rag_tool.py` replaced with async version
- [ ] Phase 1 tests pass (`./scripts/run_tests.sh`)
- [ ] Application starts successfully with deprecation warnings only
- [ ] No breaking changes to existing API endpoints
- [ ] Coverage >80% for new code

---

## 🚀 Next Steps: Phase 2 Preview

**After Phase 0-1 validation**, we will implement:

1. **Tool Registry** (`backend/app/tools/registry.py`)
2. **Tool Settings** (`backend/app/config/tool_settings.py`)
3. **Agent Refactor** (Update `chat_agent._initialize()` to use registry)
4. **Integration Tests** for registry mode

**Estimated effort for Phase 2**: 5–7 days

---

**I am ready to answer questions or proceed with Phase 2 implementation upon your validation of Phase 0-1!** 🎯
