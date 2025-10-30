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
