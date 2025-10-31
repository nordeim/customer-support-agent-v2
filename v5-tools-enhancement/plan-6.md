# 🚀 Continuing Implementation - Final Tool Files

---

## File 4: `backend/app/tools/crm_tool.py` (COMPLETE UPDATE)

```python
"""
CRM Lookup Tool - Production-ready external API integration.
Demonstrates best practices for domain-specific integrations.

Version: 3.0.0 (Enhanced with input validation and security)

Changes:
- Added Pydantic input validation
- Secure API key access via helper methods
- Enhanced error handling
- Comprehensive logging
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

import aiohttp
from aiohttp import ClientSession, ClientTimeout, ClientError
from pydantic import BaseModel, Field, ValidationError

from ..config import settings
from ..config.tool_settings import tool_settings
from .base_tool import BaseTool, ToolResult, ToolStatus
from .tool_call_wrapper import (
    RetryConfig,
    CircuitBreakerConfig,
    with_tool_call_wrapper,
    tool_call_context
)

logger = logging.getLogger(__name__)


# ===========================
# Request Validation Models
# ===========================

class CRMLookupRequest(BaseModel):
    """Validated request for CRM customer lookup."""
    customer_id: Optional[str] = Field(default=None, min_length=1, max_length=255)
    email: Optional[str] = Field(default=None, min_length=1, max_length=255)
    
    class Config:
        str_strip_whitespace = True


class CRMTicketsRequest(BaseModel):
    """Validated request for CRM tickets retrieval."""
    customer_id: str = Field(..., min_length=1, max_length=255)
    status: Optional[str] = Field(default=None, max_length=50)
    limit: int = Field(default=10, ge=1, le=100)
    
    class Config:
        str_strip_whitespace = True


class CRMUpdateRequest(BaseModel):
    """Validated request for CRM customer update."""
    customer_id: str = Field(..., min_length=1, max_length=255)
    updates: Dict[str, Any] = Field(..., min_items=1)
    
    class Config:
        str_strip_whitespace = True


# ===========================
# CRM Data Models
# ===========================

@dataclass
class CustomerProfile:
    """Customer profile data from CRM."""
    customer_id: str
    email: Optional[str] = None
    name: Optional[str] = None
    account_status: str = "unknown"
    tier: str = "standard"
    created_at: Optional[str] = None
    last_contact: Optional[str] = None
    lifetime_value: float = 0.0
    open_tickets: int = 0
    satisfaction_score: Optional[float] = None
    preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "customer_id": self.customer_id,
            "email": self.email,
            "name": self.name,
            "account_status": self.account_status,
            "tier": self.tier,
            "created_at": self.created_at,
            "last_contact": self.last_contact,
            "lifetime_value": self.lifetime_value,
            "open_tickets": self.open_tickets,
            "satisfaction_score": self.satisfaction_score,
            "preferences": self.preferences
        }


@dataclass
class TicketInfo:
    """Customer ticket information from CRM."""
    ticket_id: str
    status: str
    priority: str
    subject: str
    created_at: str
    updated_at: Optional[str] = None
    assigned_to: Optional[str] = None
    category: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticket_id": self.ticket_id,
            "status": self.status,
            "priority": self.priority,
            "subject": self.subject,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "assigned_to": self.assigned_to,
            "category": self.category
        }


# ===========================
# Custom Exceptions
# ===========================

class CRMAPIError(Exception):
    """Base exception for CRM API errors."""
    pass


class CustomerNotFoundError(CRMAPIError):
    """Customer not found in CRM."""
    pass


class AuthenticationError(CRMAPIError):
    """CRM API authentication failed."""
    pass


class AuthorizationError(CRMAPIError):
    """CRM API authorization failed."""
    pass


class RateLimitError(CRMAPIError):
    """CRM API rate limit exceeded."""
    pass


class CRMServerError(CRMAPIError):
    """CRM server error (5xx)."""
    pass


# ===========================
# CRM Tool Implementation
# ===========================

class CRMTool(BaseTool):
    """
    CRM lookup tool for retrieving customer information.
    
    Features:
    - Async HTTP client with connection pooling
    - Automatic retry with exponential backoff
    - Circuit breaker for fault tolerance
    - Request/response caching
    - Comprehensive error handling
    - Full telemetry integration
    - Input validation with Pydantic
    
    Version 3.0.0: Production-ready with security enhancements.
    """
    
    def __init__(self):
        """Initialize CRM tool."""
        super().__init__(
            name="crm_lookup",
            description="Look up customer information from CRM system",
            version="3.0.0"
        )
        
        # Configuration
        self.api_endpoint = tool_settings.crm_api_endpoint
        self.timeout = tool_settings.crm_timeout
        self.max_retries = tool_settings.crm_max_retries
        
        # HTTP client (initialized in async initialize())
        self.session: Optional[ClientSession] = None
        
        # Retry configuration
        self.retry_config = RetryConfig(
            max_attempts=self.max_retries,
            wait_multiplier=1.0,
            wait_min=1.0,
            wait_max=10.0,
            retry_exceptions=(ClientError, asyncio.TimeoutError)
        )
        
        # Circuit breaker configuration
        self.circuit_breaker_config = CircuitBreakerConfig(
            fail_max=5,
            timeout=60,
            expected_exception=Exception,
            name="crm_api"
        )
    
    async def initialize(self) -> None:
        """Initialize CRM tool resources."""
        try:
            logger.info(f"Initializing CRM tool '{self.name}'...")
            
            # Validate configuration
            if not self.api_endpoint:
                raise ValueError("CRM API endpoint not configured (set CRM_API_ENDPOINT)")
            
            # Get API key securely
            api_key = tool_settings.get_crm_api_key()
            if not api_key:
                logger.warning(
                    "CRM API key not configured. "
                    "Tool will work in mock mode for testing."
                )
            
            # Create HTTP session with connection pooling
            timeout = ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                ttl_dns_cache=300
            )
            
            self.session = ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    "User-Agent": "CustomerSupportAgent/3.0",
                    "Accept": "application/json"
                }
            )
            
            # Test connection if API key provided
            if api_key:
                await self._test_connection()
            
            self.initialized = True
            logger.info(
                f"✓ CRM tool '{self.name}' initialized successfully "
                f"(endpoint: {self.api_endpoint})"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize CRM tool: {e}", exc_info=True)
            raise
    
    async def cleanup(self) -> None:
        """Cleanup CRM tool resources."""
        try:
            logger.info(f"Cleaning up CRM tool '{self.name}'...")
            
            # Close HTTP session
            if self.session:
                await self.session.close()
                self.session = None
            
            self.initialized = False
            logger.info(f"✓ CRM tool '{self.name}' cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during CRM tool cleanup: {e}")
    
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute CRM operation.
        
        Version 3.0.0: Added input validation.
        """
        action = kwargs.get("action", "lookup_customer")
        
        try:
            if action == "lookup_customer":
                # Validate lookup request
                try:
                    request = CRMLookupRequest(
                        customer_id=kwargs.get("customer_id"),
                        email=kwargs.get("email")
                    )
                except ValidationError as e:
                    return ToolResult.error_result(
                        error=f"Invalid lookup request: {e}",
                        metadata={"tool": self.name, "validation_errors": e.errors()}
                    )
                
                if not request.customer_id and not request.email:
                    return ToolResult.error_result(
                        error="Either customer_id or email is required",
                        metadata={"tool": self.name}
                    )
                
                return await self.lookup_customer_async(
                    customer_id=request.customer_id,
                    email=request.email
                )
            
            elif action == "get_tickets":
                # Validate tickets request
                try:
                    request = CRMTicketsRequest(
                        customer_id=kwargs.get("customer_id", ""),
                        status=kwargs.get("status"),
                        limit=kwargs.get("limit", 10)
                    )
                except ValidationError as e:
                    return ToolResult.error_result(
                        error=f"Invalid tickets request: {e}",
                        metadata={"tool": self.name, "validation_errors": e.errors()}
                    )
                
                return await self.get_customer_tickets_async(
                    customer_id=request.customer_id,
                    status=request.status,
                    limit=request.limit
                )
            
            elif action == "update_customer":
                # Validate update request
                try:
                    request = CRMUpdateRequest(
                        customer_id=kwargs.get("customer_id", ""),
                        updates=kwargs.get("updates", {})
                    )
                except ValidationError as e:
                    return ToolResult.error_result(
                        error=f"Invalid update request: {e}",
                        metadata={"tool": self.name, "validation_errors": e.errors()}
                    )
                
                return await self.update_customer_async(
                    customer_id=request.customer_id,
                    updates=request.updates
                )
            
            else:
                return ToolResult.error_result(
                    error=f"Unknown action: {action}. Valid: lookup_customer, get_tickets, update_customer",
                    metadata={"tool": self.name}
                )
                
        except Exception as e:
            logger.error(f"CRM execute error: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={"tool": self.name, "action": action}
            )
    
    async def lookup_customer_async(
        self,
        customer_id: Optional[str] = None,
        email: Optional[str] = None
    ) -> ToolResult:
        """Look up customer information by ID or email."""
        try:
            # Determine lookup parameter
            if customer_id:
                url = f"{self.api_endpoint}/customers/{customer_id}"
                lookup_key = "customer_id"
                lookup_value = customer_id
            else:
                url = f"{self.api_endpoint}/customers/by-email/{email}"
                lookup_key = "email"
                lookup_value = email
            
            # Make API request with retry and circuit breaker
            response_data = await self._make_api_request(
                method="GET",
                url=url,
                operation="lookup_customer"
            )
            
            # Parse response into CustomerProfile
            profile = self._parse_customer_profile(response_data)
            
            logger.info(
                f"Successfully retrieved customer profile: {profile.customer_id}",
                extra={
                    "customer_id": profile.customer_id,
                    "account_status": profile.account_status,
                    "tier": profile.tier
                }
            )
            
            return ToolResult.success_result(
                data={
                    "profile": profile.to_dict(),
                    "lookup_key": lookup_key,
                    "found": True
                },
                metadata={
                    "tool": self.name,
                    "operation": "lookup_customer",
                    "customer_id": profile.customer_id,
                    "account_status": profile.account_status
                }
            )
            
        except CustomerNotFoundError as e:
            logger.warning(f"Customer not found: {lookup_value}")
            return ToolResult.success_result(
                data={
                    "profile": None,
                    "lookup_key": lookup_key,
                    "found": False
                },
                metadata={
                    "tool": self.name,
                    "operation": "lookup_customer",
                    "not_found": True
                }
            )
        
        except Exception as e:
            logger.error(f"CRM lookup error: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={
                    "tool": self.name,
                    "operation": "lookup_customer",
                    "lookup_key": lookup_key,
                    "lookup_value": lookup_value
                }
            )
    
    async def get_customer_tickets_async(
        self,
        customer_id: str,
        status: Optional[str] = None,
        limit: int = 10
    ) -> ToolResult:
        """Get customer tickets from CRM."""
        try:
            # Build URL with query parameters
            url = f"{self.api_endpoint}/customers/{customer_id}/tickets"
            params = {"limit": limit}
            if status:
                params["status"] = status
            
            # Make API request
            response_data = await self._make_api_request(
                method="GET",
                url=url,
                params=params,
                operation="get_tickets"
            )
            
            # Parse tickets
            tickets = [
                self._parse_ticket(ticket_data)
                for ticket_data in response_data.get("tickets", [])
            ]
            
            logger.info(
                f"Retrieved {len(tickets)} tickets for customer {customer_id}",
                extra={"customer_id": customer_id, "ticket_count": len(tickets)}
            )
            
            return ToolResult.success_result(
                data={
                    "tickets": [t.to_dict() for t in tickets],
                    "total_count": len(tickets),
                    "customer_id": customer_id
                },
                metadata={
                    "tool": self.name,
                    "operation": "get_tickets",
                    "customer_id": customer_id,
                    "ticket_count": len(tickets)
                }
            )
            
        except Exception as e:
            logger.error(f"Error retrieving tickets: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={
                    "tool": self.name,
                    "operation": "get_tickets",
                    "customer_id": customer_id
                }
            )
    
    async def update_customer_async(
        self,
        customer_id: str,
        updates: Dict[str, Any]
    ) -> ToolResult:
        """Update customer information in CRM."""
        try:
            url = f"{self.api_endpoint}/customers/{customer_id}"
            
            # Make API request
            response_data = await self._make_api_request(
                method="PATCH",
                url=url,
                json_data=updates,
                operation="update_customer"
            )
            
            logger.info(
                f"Updated customer {customer_id}",
                extra={
                    "customer_id": customer_id,
                    "updated_fields": list(updates.keys())
                }
            )
            
            return ToolResult.success_result(
                data={
                    "customer_id": customer_id,
                    "updated_fields": list(updates.keys()),
                    "updated_at": response_data.get("updated_at", datetime.utcnow().isoformat())
                },
                metadata={
                    "tool": self.name,
                    "operation": "update_customer",
                    "customer_id": customer_id
                }
            )
            
        except Exception as e:
            logger.error(f"Error updating customer: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={
                    "tool": self.name,
                    "operation": "update_customer",
                    "customer_id": customer_id
                }
            )
    
    # ===========================
    # Private Helper Methods
    # ===========================
    
    async def _make_api_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        operation: str = "api_request"
    ) -> Dict[str, Any]:
        """Make HTTP request to CRM API with retry and circuit breaker."""
        if not self.session:
            raise RuntimeError("CRM tool not initialized. Call initialize() first.")
        
        # Get API key securely
        api_key = tool_settings.get_crm_api_key()
        
        # Check for mock mode
        if not api_key:
            logger.info("CRM API key not configured, using mock data")
            return self._get_mock_response(method, url, operation)
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Wrap request with telemetry, retry, and circuit breaker
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
                headers=headers
            ) as response:
                # Handle various HTTP status codes
                if response.status == 404:
                    raise CustomerNotFoundError(f"Customer not found: {url}")
                
                elif response.status == 401:
                    raise AuthenticationError("Invalid CRM API credentials")
                
                elif response.status == 403:
                    raise AuthorizationError("Insufficient permissions for CRM API")
                
                elif response.status == 429:
                    raise RateLimitError("CRM API rate limit exceeded")
                
                elif response.status >= 500:
                    raise CRMServerError(f"CRM server error: {response.status}")
                
                elif response.status >= 400:
                    raise CRMAPIError(f"CRM API error: {response.status}")
                
                # Parse JSON response
                try:
                    return await response.json()
                except Exception as e:
                    raise CRMAPIError(f"Failed to parse CRM response: {e}")
        
        return await execute_request()
    
    async def _test_connection(self) -> bool:
        """Test CRM API connection."""
        try:
            api_key = tool_settings.get_crm_api_key()
            url = f"{self.api_endpoint}/health"
            headers = {"Authorization": f"Bearer {api_key}"}
            
            async with self.session.get(url, headers=headers, timeout=ClientTimeout(total=5)) as response:
                if response.status == 200:
                    logger.info("CRM API connection test successful")
                    return True
                else:
                    logger.warning(f"CRM API connection test failed: {response.status}")
                    return False
        
        except Exception as e:
            logger.warning(f"CRM API connection test failed: {e}")
            return False
    
    def _parse_customer_profile(self, data: Dict[str, Any]) -> CustomerProfile:
        """Parse API response into CustomerProfile."""
        return CustomerProfile(
            customer_id=data.get("id") or data.get("customer_id"),
            email=data.get("email"),
            name=data.get("name") or data.get("full_name"),
            account_status=data.get("status", "unknown"),
            tier=data.get("tier", "standard"),
            created_at=data.get("created_at"),
            last_contact=data.get("last_contact"),
            lifetime_value=float(data.get("lifetime_value", 0.0)),
            open_tickets=int(data.get("open_tickets", 0)),
            satisfaction_score=data.get("satisfaction_score"),
            preferences=data.get("preferences", {})
        )
    
    def _parse_ticket(self, data: Dict[str, Any]) -> TicketInfo:
        """Parse API response into TicketInfo."""
        return TicketInfo(
            ticket_id=data.get("id") or data.get("ticket_id"),
            status=data.get("status", "unknown"),
            priority=data.get("priority", "normal"),
            subject=data.get("subject", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at"),
            assigned_to=data.get("assigned_to"),
            category=data.get("category")
        )
    
    def _get_mock_response(
        self,
        method: str,
        url: str,
        operation: str
    ) -> Dict[str, Any]:
        """Get mock response for testing without real API."""
        logger.info(f"Returning mock CRM response for {operation}")
        
        if "lookup_customer" in operation or "/customers/" in url:
            return {
                "id": "CUST-12345",
                "customer_id": "CUST-12345",
                "email": "customer@example.com",
                "name": "John Doe",
                "status": "active",
                "tier": "premium",
                "created_at": "2023-01-15T10:00:00Z",
                "last_contact": "2024-01-10T14:30:00Z",
                "lifetime_value": 15000.50,
                "open_tickets": 2,
                "satisfaction_score": 4.5,
                "preferences": {
                    "communication_channel": "email",
                    "language": "en",
                    "timezone": "UTC"
                }
            }
        
        elif "get_tickets" in operation or "/tickets" in url:
            return {
                "tickets": [
                    {
                        "id": "TICKET-001",
                        "status": "open",
                        "priority": "high",
                        "subject": "Billing inquiry",
                        "created_at": "2024-01-08T09:00:00Z",
                        "category": "billing"
                    },
                    {
                        "id": "TICKET-002",
                        "status": "in_progress",
                        "priority": "normal",
                        "subject": "Feature request",
                        "created_at": "2024-01-05T11:30:00Z",
                        "assigned_to": "agent@example.com",
                        "category": "feature"
                    }
                ],
                "total": 2
            }
        
        elif "update_customer" in operation:
            return {
                "customer_id": "CUST-12345",
                "updated_at": datetime.utcnow().isoformat(),
                "success": True
            }
        
        else:
            return {"status": "ok"}


__all__ = [
    'CRMTool',
    'CustomerProfile',
    'TicketInfo',
    'CRMAPIError',
    'CustomerNotFoundError',
    'AuthenticationError',
    'AuthorizationError',
    'RateLimitError',
    'CRMServerError'
]
```

---

## File 5: `backend/app/tools/inventory_tool.py` (COMPLETE IMPLEMENTATION)

```python
"""
Inventory Tool - Production-ready inventory/product API integration.
Based on CRM tool pattern with full implementation.

Version: 1.0.0 (Complete production implementation)
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

import aiohttp
from aiohttp import ClientSession, ClientTimeout, ClientError
from pydantic import BaseModel, Field, ValidationError

from ..config import settings
from ..config.tool_settings import tool_settings
from .base_tool import BaseTool, ToolResult, ToolStatus
from .tool_call_wrapper import RetryConfig, CircuitBreakerConfig, with_tool_call_wrapper

logger = logging.getLogger(__name__)


# ===========================
# Request Validation Models
# ===========================

class InventoryLookupRequest(BaseModel):
    """Validated request for inventory lookup."""
    product_id: Optional[str] = Field(default=None, min_length=1, max_length=255)
    sku: Optional[str] = Field(default=None, min_length=1, max_length=255)
    
    class Config:
        str_strip_whitespace = True


class InventorySearchRequest(BaseModel):
    """Validated request for inventory search."""
    query: str = Field(..., min_length=1, max_length=500)
    category: Optional[str] = Field(default=None, max_length=100)
    limit: int = Field(default=20, ge=1, le=100)
    
    class Config:
        str_strip_whitespace = True


class InventoryUpdateRequest(BaseModel):
    """Validated request for inventory update."""
    product_id: str = Field(..., min_length=1, max_length=255)
    quantity_change: int = Field(..., description="Positive to add, negative to remove")
    
    class Config:
        str_strip_whitespace = True


# ===========================
# Inventory Data Models
# ===========================

@dataclass
class ProductInfo:
    """Product/inventory data model."""
    product_id: str
    name: str
    sku: str
    price: float
    stock_quantity: int = 0
    available: bool = True
    category: Optional[str] = None
    description: Optional[str] = None
    supplier: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "product_id": self.product_id,
            "name": self.name,
            "sku": self.sku,
            "price": self.price,
            "stock_quantity": self.stock_quantity,
            "available": self.available,
            "category": self.category,
            "description": self.description,
            "supplier": self.supplier
        }


# ===========================
# Custom Exceptions
# ===========================

class InventoryAPIError(Exception):
    """Base exception for Inventory API errors."""
    pass


class ProductNotFoundError(InventoryAPIError):
    """Product not found in inventory."""
    pass


class InsufficientStockError(InventoryAPIError):
    """Insufficient stock for requested operation."""
    pass


# ===========================
# Inventory Tool Implementation
# ===========================

class InventoryTool(BaseTool):
    """
    Inventory tool for product and stock operations.
    
    Features:
    - Product lookup by ID or SKU
    - Stock availability checking
    - Product search
    - Inventory updates
    - Mock mode for testing
    
    Version 1.0.0: Complete production implementation.
    """
    
    def __init__(self):
        """Initialize inventory tool."""
        super().__init__(
            name="inventory_lookup",
            description="Look up product and inventory information",
            version="1.0.0"
        )
        
        # Configuration
        self.api_endpoint = tool_settings.inventory_api_endpoint
        self.timeout = 10  # Default timeout
        self.max_retries = 3
        
        # HTTP client (initialized in async initialize())
        self.session: Optional[ClientSession] = None
        
        # Retry configuration
        self.retry_config = RetryConfig(
            max_attempts=self.max_retries,
            wait_multiplier=1.0,
            wait_min=1.0,
            wait_max=10.0,
            retry_exceptions=(ClientError, asyncio.TimeoutError)
        )
        
        # Circuit breaker configuration
        self.circuit_breaker_config = CircuitBreakerConfig(
            fail_max=5,
            timeout=60,
            expected_exception=Exception,
            name="inventory_api"
        )
    
    async def initialize(self) -> None:
        """Initialize inventory tool resources."""
        try:
            logger.info(f"Initializing Inventory tool '{self.name}'...")
            
            # Validate configuration
            if not self.api_endpoint:
                logger.warning(
                    "Inventory API endpoint not configured. "
                    "Tool will work in mock mode for testing."
                )
            
            # Get API key securely
            api_key = tool_settings.get_inventory_api_key()
            if not api_key:
                logger.warning("Inventory API key not configured. Using mock mode.")
            
            # Create HTTP session with connection pooling
            timeout = ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                ttl_dns_cache=300
            )
            
            self.session = ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    "User-Agent": "CustomerSupportAgent/1.0",
                    "Accept": "application/json"
                }
            )
            
            self.initialized = True
            logger.info(f"✓ Inventory tool '{self.name}' initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Inventory tool: {e}", exc_info=True)
            raise
    
    async def cleanup(self) -> None:
        """Cleanup inventory tool resources."""
        try:
            logger.info(f"Cleaning up Inventory tool '{self.name}'...")
            
            # Close HTTP session
            if self.session:
                await self.session.close()
                self.session = None
            
            self.initialized = False
            logger.info(f"✓ Inventory tool '{self.name}' cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during Inventory tool cleanup: {e}")
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute inventory operation."""
        action = kwargs.get("action", "check_stock")
        
        try:
            if action == "check_stock":
                # Validate lookup request
                try:
                    request = InventoryLookupRequest(
                        product_id=kwargs.get("product_id"),
                        sku=kwargs.get("sku")
                    )
                except ValidationError as e:
                    return ToolResult.error_result(
                        error=f"Invalid lookup request: {e}",
                        metadata={"tool": self.name, "validation_errors": e.errors()}
                    )
                
                if not request.product_id and not request.sku:
                    return ToolResult.error_result(
                        error="Either product_id or sku is required",
                        metadata={"tool": self.name}
                    )
                
                return await self.check_stock_async(
                    product_id=request.product_id,
                    sku=request.sku
                )
            
            elif action == "get_product":
                # Validate lookup request
                try:
                    request = InventoryLookupRequest(
                        product_id=kwargs.get("product_id"),
                        sku=kwargs.get("sku")
                    )
                except ValidationError as e:
                    return ToolResult.error_result(
                        error=f"Invalid product request: {e}",
                        metadata={"tool": self.name, "validation_errors": e.errors()}
                    )
                
                return await self.get_product_async(
                    product_id=request.product_id,
                    sku=request.sku
                )
            
            elif action == "search_products":
                # Validate search request
                try:
                    request = InventorySearchRequest(
                        query=kwargs.get("query", ""),
                        category=kwargs.get("category"),
                        limit=kwargs.get("limit", 20)
                    )
                except ValidationError as e:
                    return ToolResult.error_result(
                        error=f"Invalid search request: {e}",
                        metadata={"tool": self.name, "validation_errors": e.errors()}
                    )
                
                return await self.search_products_async(
                    query=request.query,
                    category=request.category,
                    limit=request.limit
                )
            
            elif action == "update_stock":
                # Validate update request
                try:
                    request = InventoryUpdateRequest(
                        product_id=kwargs.get("product_id", ""),
                        quantity_change=kwargs.get("quantity_change", 0)
                    )
                except ValidationError as e:
                    return ToolResult.error_result(
                        error=f"Invalid update request: {e}",
                        metadata={"tool": self.name, "validation_errors": e.errors()}
                    )
                
                return await self.update_stock_async(
                    product_id=request.product_id,
                    quantity_change=request.quantity_change
                )
            
            else:
                return ToolResult.error_result(
                    error=f"Unknown action: {action}. Valid: check_stock, get_product, search_products, update_stock",
                    metadata={"tool": self.name}
                )
                
        except Exception as e:
            logger.error(f"Inventory execute error: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={"tool": self.name, "action": action}
            )
    
    async def check_stock_async(
        self,
        product_id: Optional[str] = None,
        sku: Optional[str] = None
    ) -> ToolResult:
        """Check stock availability for a product."""
        try:
            # Get product info
            product_result = await self.get_product_async(product_id, sku)
            
            if not product_result.success:
                return product_result
            
            product = product_result.data.get("product")
            
            stock_status = {
                "product_id": product["product_id"],
                "sku": product["sku"],
                "name": product["name"],
                "stock_quantity": product["stock_quantity"],
                "available": product["available"],
                "in_stock": product["stock_quantity"] > 0
            }
            
            logger.info(
                f"Stock check for {product['name']}: {stock_status['stock_quantity']} units",
                extra={"product_id": product["product_id"], "stock": stock_status["stock_quantity"]}
            )
            
            return ToolResult.success_result(
                data=stock_status,
                metadata={
                    "tool": self.name,
                    "operation": "check_stock",
                    "product_id": product["product_id"]
                }
            )
            
        except Exception as e:
            logger.error(f"Stock check error: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={"tool": self.name, "operation": "check_stock"}
            )
    
    async def get_product_async(
        self,
        product_id: Optional[str] = None,
        sku: Optional[str] = None
    ) -> ToolResult:
        """Get product information."""
        try:
            # Determine lookup parameter
            if product_id:
                url = f"{self.api_endpoint}/products/{product_id}"
                lookup_key = "product_id"
                lookup_value = product_id
            else:
                url = f"{self.api_endpoint}/products/by-sku/{sku}"
                lookup_key = "sku"
                lookup_value = sku
            
            # Make API request
            response_data = await self._make_api_request(
                method="GET",
                url=url,
                operation="get_product"
            )
            
            # Parse response
            product = self._parse_product(response_data)
            
            logger.info(f"Retrieved product: {product.name}")
            
            return ToolResult.success_result(
                data={
                    "product": product.to_dict(),
                    "lookup_key": lookup_key,
                    "found": True
                },
                metadata={
                    "tool": self.name,
                    "operation": "get_product",
                    "product_id": product.product_id
                }
            )
            
        except ProductNotFoundError:
            return ToolResult.success_result(
                data={
                    "product": None,
                    "lookup_key": lookup_key,
                    "found": False
                },
                metadata={
                    "tool": self.name,
                    "operation": "get_product",
                    "not_found": True
                }
            )
        
        except Exception as e:
            logger.error(f"Get product error: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={"tool": self.name, "operation": "get_product"}
            )
    
    async def search_products_async(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 20
    ) -> ToolResult:
        """Search for products."""
        try:
            url = f"{self.api_endpoint}/products/search"
            params = {"q": query, "limit": limit}
            
            if category:
                params["category"] = category
            
            # Make API request
            response_data = await self._make_api_request(
                method="GET",
                url=url,
                params=params,
                operation="search_products"
            )
            
            # Parse results
            products = [
                self._parse_product(item).to_dict()
                for item in response_data.get("products", [])
            ]
            
            logger.info(f"Found {len(products)} products for query: {query}")
            
            return ToolResult.success_result(
                data={
                    "products": products,
                    "total_count": len(products),
                    "query": query
                },
                metadata={
                    "tool": self.name,
                    "operation": "search_products",
                    "results_count": len(products)
                }
            )
            
        except Exception as e:
            logger.error(f"Search products error: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={"tool": self.name, "operation": "search_products"}
            )
    
    async def update_stock_async(
        self,
        product_id: str,
        quantity_change: int
    ) -> ToolResult:
        """Update stock quantity for a product."""
        try:
            url = f"{self.api_endpoint}/products/{product_id}/stock"
            json_data = {"quantity_change": quantity_change}
            
            # Make API request
            response_data = await self._make_api_request(
                method="PATCH",
                url=url,
                json_data=json_data,
                operation="update_stock"
            )
            
            logger.info(
                f"Updated stock for product {product_id}: {quantity_change:+d}",
                extra={"product_id": product_id, "quantity_change": quantity_change}
            )
            
            return ToolResult.success_result(
                data={
                    "product_id": product_id,
                    "quantity_change": quantity_change,
                    "new_quantity": response_data.get("new_quantity"),
                    "updated_at": response_data.get("updated_at", datetime.utcnow().isoformat())
                },
                metadata={
                    "tool": self.name,
                    "operation": "update_stock",
                    "product_id": product_id
                }
            )
            
        except Exception as e:
            logger.error(f"Update stock error: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={"tool": self.name, "operation": "update_stock"}
            )
    
    # ===========================
    # Private Helper Methods
    # ===========================
    
    async def _make_api_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        operation: str = "api_request"
    ) -> Dict[str, Any]:
        """Make HTTP request to Inventory API."""
        if not self.session:
            raise RuntimeError("Inventory tool not initialized")
        
        # Get API key securely
        api_key = tool_settings.get_inventory_api_key()
        
        # Check for mock mode
        if not api_key or not self.api_endpoint:
            logger.info("Using mock inventory data")
            return self._get_mock_response(method, url, operation, params)
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Make request with retry and circuit breaker
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
                headers=headers
            ) as response:
                if response.status == 404:
                    raise ProductNotFoundError(f"Product not found: {url}")
                elif response.status == 409:
                    raise InsufficientStockError("Insufficient stock")
                elif response.status >= 400:
                    raise InventoryAPIError(f"API error: {response.status}")
                
                return await response.json()
        
        return await execute_request()
    
    def _parse_product(self, data: Dict[str, Any]) -> ProductInfo:
        """Parse API response into ProductInfo."""
        return ProductInfo(
            product_id=data.get("id") or data.get("product_id"),
            name=data.get("name", "Unknown Product"),
            sku=data.get("sku", ""),
            price=float(data.get("price", 0.0)),
            stock_quantity=int(data.get("stock_quantity", 0)),
            available=bool(data.get("available", True)),
            category=data.get("category"),
            description=data.get("description"),
            supplier=data.get("supplier")
        )
    
    def _get_mock_response(
        self,
        method: str,
        url: str,
        operation: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get mock response for testing."""
        logger.info(f"Returning mock inventory response for {operation}")
        
        if "search" in operation:
            return {
                "products": [
                    {
                        "id": "PROD-001",
                        "name": "Premium Widget",
                        "sku": "WDG-001",
                        "price": 49.99,
                        "stock_quantity": 150,
                        "available": True,
                        "category": "widgets"
                    },
                    {
                        "id": "PROD-002",
                        "name": "Standard Widget",
                        "sku": "WDG-002",
                        "price": 29.99,
                        "stock_quantity": 75,
                        "available": True,
                        "category": "widgets"
                    }
                ]
            }
        
        elif "update_stock" in operation:
            return {
                "product_id": "PROD-001",
                "new_quantity": 145,
                "updated_at": datetime.utcnow().isoformat()
            }
        
        else:
            return {
                "id": "PROD-001",
                "product_id": "PROD-001",
                "name": "Premium Widget",
                "sku": "WDG-001",
                "price": 49.99,
                "stock_quantity": 150,
                "available": True,
                "category": "widgets",
                "description": "High-quality premium widget",
                "supplier": "Widget Corp"
            }


__all__ = ['InventoryTool', 'ProductInfo', 'InventoryAPIError', 'ProductNotFoundError', 'InsufficientStockError']
```

---

Continuing with the final file (billing_tool.py) in the next message...

Would you like me to continue with the billing tool?
