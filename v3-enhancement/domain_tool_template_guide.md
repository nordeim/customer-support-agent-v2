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
