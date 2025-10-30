"""
CRM Lookup Tool - Domain integration template.
Demonstrates production-ready external API integration with retry, circuit breaker, and telemetry.

Phase 5: Reference implementation for domain-specific tools.
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

import aiohttp
from aiohttp import ClientSession, ClientTimeout, ClientError

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
    
    Phase 5: Serves as template for all domain integrations.
    """
    
    def __init__(self):
        """Initialize CRM tool."""
        # Call parent init (no auto-initialization in Phase 1+)
        super().__init__(
            name="crm_lookup",
            description="Look up customer information from CRM system"
        )
        
        # Configuration
        self.api_endpoint = tool_settings.crm_api_endpoint
        self.api_key = tool_settings.crm_api_key
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
    
    # ===========================
    # Async Interface (Phase 1+)
    # ===========================
    
    async def initialize(self) -> None:
        """
        Initialize CRM tool resources.
        Sets up HTTP client with connection pooling.
        """
        try:
            logger.info(f"Initializing CRM tool '{self.name}'...")
            
            # Validate configuration
            if not self.api_endpoint:
                raise ValueError("CRM API endpoint not configured (CRM_API_ENDPOINT)")
            
            if not self.api_key:
                logger.warning(
                    "CRM API key not configured. "
                    "Tool will work in mock mode for testing."
                )
            
            # Create HTTP session with connection pooling
            timeout = ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(
                limit=10,  # Max concurrent connections
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
            
            # Test connection if API key provided
            if self.api_key:
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
        
        Accepts:
            action: Operation to perform ('lookup_customer', 'get_tickets', 'update_customer')
            customer_id: Customer identifier (required for lookup/get_tickets)
            email: Customer email (alternative lookup key)
            updates: Update data (for update_customer)
            
        Returns:
            ToolResult with operation results
        """
        action = kwargs.get("action", "lookup_customer")
        
        if action == "lookup_customer":
            return await self.lookup_customer_async(
                customer_id=kwargs.get("customer_id"),
                email=kwargs.get("email")
            )
        
        elif action == "get_tickets":
            return await self.get_customer_tickets_async(
                customer_id=kwargs.get("customer_id")
            )
        
        elif action == "update_customer":
            return await self.update_customer_async(
                customer_id=kwargs.get("customer_id"),
                updates=kwargs.get("updates", {})
            )
        
        else:
            return ToolResult.error_result(
                error=f"Unknown action: {action}. Valid actions: lookup_customer, get_tickets, update_customer",
                metadata={"tool": self.name}
            )
    
    # ===========================
    # Core CRM Methods (Async)
    # ===========================
    
    async def lookup_customer_async(
        self,
        customer_id: Optional[str] = None,
        email: Optional[str] = None
    ) -> ToolResult:
        """
        Look up customer information by ID or email.
        
        Args:
            customer_id: Customer ID
            email: Customer email (alternative lookup)
            
        Returns:
            ToolResult with CustomerProfile
        """
        if not customer_id and not email:
            return ToolResult.error_result(
                error="Either customer_id or email is required",
                metadata={"tool": self.name}
            )
        
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
        """
        Get customer tickets from CRM.
        
        Args:
            customer_id: Customer ID
            status: Filter by status (optional)
            limit: Maximum tickets to retrieve
            
        Returns:
            ToolResult with list of tickets
        """
        if not customer_id:
            return ToolResult.error_result(
                error="customer_id is required",
                metadata={"tool": self.name}
            )
        
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
        """
        Update customer information in CRM.
        
        Args:
            customer_id: Customer ID
            updates: Fields to update
            
        Returns:
            ToolResult with update status
        """
        if not customer_id:
            return ToolResult.error_result(
                error="customer_id is required",
                metadata={"tool": self.name}
            )
        
        if not updates:
            return ToolResult.error_result(
                error="updates dictionary cannot be empty",
                metadata={"tool": self.name}
            )
        
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
    # Legacy Methods (Backward Compatibility)
    # ===========================
    
    def _setup(self) -> None:
        """
        DEPRECATED: Legacy sync setup.
        Use async initialize() instead.
        """
        logger.warning("CRMTool._setup is deprecated. Use await crm_tool.initialize()")
        raise NotImplementedError(
            "CRMTool requires async initialization. "
            "Call await crm_tool.initialize() instead."
        )
    
    async def lookup_customer(
        self,
        customer_id: Optional[str] = None,
        email: Optional[str] = None
    ) -> ToolResult:
        """Legacy method (already async, kept for compatibility)."""
        return await self.lookup_customer_async(customer_id, email)
    
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
        """
        Make HTTP request to CRM API with retry and circuit breaker.
        
        Args:
            method: HTTP method
            url: Request URL
            params: Query parameters
            json_data: JSON request body
            operation: Operation name for telemetry
            
        Returns:
            Response data as dictionary
            
        Raises:
            Various exceptions for different error conditions
        """
        if not self.session:
            raise RuntimeError("CRM tool not initialized. Call initialize() first.")
        
        # Check for mock mode
        if not self.api_key:
            logger.info("CRM API key not configured, using mock data")
            return self._get_mock_response(method, url, operation)
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
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
        """
        Test CRM API connection.
        
        Returns:
            True if connection successful
        """
        try:
            url = f"{self.api_endpoint}/health"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
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
        """
        Parse API response into CustomerProfile.
        
        Args:
            data: Response data from API
            
        Returns:
            CustomerProfile instance
        """
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
        """
        Parse API response into TicketInfo.
        
        Args:
            data: Ticket data from API
            
        Returns:
            TicketInfo instance
        """
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
        """
        Get mock response for testing without real API.
        
        Args:
            method: HTTP method
            url: Request URL
            operation: Operation name
            
        Returns:
            Mock response data
        """
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


# Export
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
