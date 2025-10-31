"""
Billing Tool - Production-ready billing/invoice API integration.
Based on CRM tool pattern with complete implementation.

Version: 1.0.0 (Complete production implementation)
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
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

class BillingInvoicesRequest(BaseModel):
    """Validated request for retrieving invoices."""
    customer_id: str = Field(..., min_length=1, max_length=255)
    status: Optional[str] = Field(default=None, max_length=50)
    limit: int = Field(default=10, ge=1, le=100)
    days_back: Optional[int] = Field(default=None, ge=1, le=365)
    
    class Config:
        str_strip_whitespace = True


class BillingInvoiceRequest(BaseModel):
    """Validated request for retrieving a specific invoice."""
    invoice_id: str = Field(..., min_length=1, max_length=255)
    
    class Config:
        str_strip_whitespace = True


class BillingPaymentRequest(BaseModel):
    """Validated request for payment processing."""
    invoice_id: str = Field(..., min_length=1, max_length=255)
    amount: float = Field(..., gt=0.0)
    payment_method: str = Field(..., min_length=1, max_length=50)
    
    class Config:
        str_strip_whitespace = True


class BillingBalanceRequest(BaseModel):
    """Validated request for account balance."""
    customer_id: str = Field(..., min_length=1, max_length=255)
    
    class Config:
        str_strip_whitespace = True


# ===========================
# Billing Data Models
# ===========================

@dataclass
class Invoice:
    """Invoice data model."""
    invoice_id: str
    customer_id: str
    amount: float
    currency: str = "USD"
    status: str = "pending"
    due_date: Optional[str] = None
    created_at: Optional[str] = None
    paid_at: Optional[str] = None
    line_items: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.line_items is None:
            self.line_items = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "invoice_id": self.invoice_id,
            "customer_id": self.customer_id,
            "amount": self.amount,
            "currency": self.currency,
            "status": self.status,
            "due_date": self.due_date,
            "created_at": self.created_at,
            "paid_at": self.paid_at,
            "line_items": self.line_items
        }
    
    @property
    def is_overdue(self) -> bool:
        """Check if invoice is overdue."""
        if not self.due_date or self.status == "paid":
            return False
        
        try:
            due = datetime.fromisoformat(self.due_date.replace('Z', '+00:00'))
            return datetime.utcnow() > due
        except Exception:
            return False


@dataclass
class Payment:
    """Payment data model."""
    payment_id: str
    invoice_id: str
    amount: float
    currency: str = "USD"
    status: str = "pending"
    payment_method: str = "unknown"
    created_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "payment_id": self.payment_id,
            "invoice_id": self.invoice_id,
            "amount": self.amount,
            "currency": self.currency,
            "status": self.status,
            "payment_method": self.payment_method,
            "created_at": self.created_at
        }


@dataclass
class AccountBalance:
    """Account balance data model."""
    customer_id: str
    current_balance: float
    outstanding_invoices: int
    total_outstanding: float
    currency: str = "USD"
    last_payment: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "customer_id": self.customer_id,
            "current_balance": self.current_balance,
            "outstanding_invoices": self.outstanding_invoices,
            "total_outstanding": self.total_outstanding,
            "currency": self.currency,
            "last_payment": self.last_payment
        }


# ===========================
# Custom Exceptions
# ===========================

class BillingAPIError(Exception):
    """Base exception for Billing API errors."""
    pass


class InvoiceNotFoundError(BillingAPIError):
    """Invoice not found in billing system."""
    pass


class PaymentFailedError(BillingAPIError):
    """Payment processing failed."""
    pass


class InsufficientFundsError(BillingAPIError):
    """Insufficient funds for payment."""
    pass


# ===========================
# Billing Tool Implementation
# ===========================

class BillingTool(BaseTool):
    """
    Billing tool for invoice and payment operations.
    
    Features:
    - Invoice retrieval and management
    - Payment processing
    - Account balance checking
    - Payment history
    - Mock mode for testing
    
    Version 1.0.0: Complete production implementation.
    """
    
    def __init__(self):
        """Initialize billing tool."""
        super().__init__(
            name="billing_lookup",
            description="Look up billing and invoice information",
            version="1.0.0"
        )
        
        # Configuration
        self.api_endpoint = tool_settings.billing_api_endpoint
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
            name="billing_api"
        )
    
    async def initialize(self) -> None:
        """Initialize billing tool resources."""
        try:
            logger.info(f"Initializing Billing tool '{self.name}'...")
            
            # Validate configuration
            if not self.api_endpoint:
                logger.warning(
                    "Billing API endpoint not configured. "
                    "Tool will work in mock mode for testing."
                )
            
            # Get API key securely
            api_key = tool_settings.get_billing_api_key()
            if not api_key:
                logger.warning("Billing API key not configured. Using mock mode.")
            
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
            logger.info(f"✓ Billing tool '{self.name}' initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Billing tool: {e}", exc_info=True)
            raise
    
    async def cleanup(self) -> None:
        """Cleanup billing tool resources."""
        try:
            logger.info(f"Cleaning up Billing tool '{self.name}'...")
            
            # Close HTTP session
            if self.session:
                await self.session.close()
                self.session = None
            
            self.initialized = False
            logger.info(f"✓ Billing tool '{self.name}' cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during Billing tool cleanup: {e}")
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute billing operation."""
        action = kwargs.get("action", "get_invoices")
        
        try:
            if action == "get_invoices":
                # Validate invoices request
                try:
                    request = BillingInvoicesRequest(
                        customer_id=kwargs.get("customer_id", ""),
                        status=kwargs.get("status"),
                        limit=kwargs.get("limit", 10),
                        days_back=kwargs.get("days_back")
                    )
                except ValidationError as e:
                    return ToolResult.error_result(
                        error=f"Invalid invoices request: {e}",
                        metadata={"tool": self.name, "validation_errors": e.errors()}
                    )
                
                return await self.get_invoices_async(
                    customer_id=request.customer_id,
                    status=request.status,
                    limit=request.limit,
                    days_back=request.days_back
                )
            
            elif action == "get_invoice":
                # Validate invoice request
                try:
                    request = BillingInvoiceRequest(
                        invoice_id=kwargs.get("invoice_id", "")
                    )
                except ValidationError as e:
                    return ToolResult.error_result(
                        error=f"Invalid invoice request: {e}",
                        metadata={"tool": self.name, "validation_errors": e.errors()}
                    )
                
                return await self.get_invoice_async(
                    invoice_id=request.invoice_id
                )
            
            elif action == "get_balance":
                # Validate balance request
                try:
                    request = BillingBalanceRequest(
                        customer_id=kwargs.get("customer_id", "")
                    )
                except ValidationError as e:
                    return ToolResult.error_result(
                        error=f"Invalid balance request: {e}",
                        metadata={"tool": self.name, "validation_errors": e.errors()}
                    )
                
                return await self.get_balance_async(
                    customer_id=request.customer_id
                )
            
            elif action == "create_payment":
                # Validate payment request
                try:
                    request = BillingPaymentRequest(
                        invoice_id=kwargs.get("invoice_id", ""),
                        amount=kwargs.get("amount", 0.0),
                        payment_method=kwargs.get("payment_method", "")
                    )
                except ValidationError as e:
                    return ToolResult.error_result(
                        error=f"Invalid payment request: {e}",
                        metadata={"tool": self.name, "validation_errors": e.errors()}
                    )
                
                return await self.create_payment_async(
                    invoice_id=request.invoice_id,
                    amount=request.amount,
                    payment_method=request.payment_method
                )
            
            else:
                return ToolResult.error_result(
                    error=f"Unknown action: {action}. Valid: get_invoices, get_invoice, get_balance, create_payment",
                    metadata={"tool": self.name}
                )
                
        except Exception as e:
            logger.error(f"Billing execute error: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={"tool": self.name, "action": action}
            )
    
    async def get_invoices_async(
        self,
        customer_id: str,
        status: Optional[str] = None,
        limit: int = 10,
        days_back: Optional[int] = None
    ) -> ToolResult:
        """Get invoices for a customer."""
        try:
            url = f"{self.api_endpoint}/customers/{customer_id}/invoices"
            params = {"limit": limit}
            
            if status:
                params["status"] = status
            if days_back:
                params["days_back"] = days_back
            
            # Make API request
            response_data = await self._make_api_request(
                method="GET",
                url=url,
                params=params,
                operation="get_invoices"
            )
            
            # Parse invoices
            invoices = [
                self._parse_invoice(inv_data).to_dict()
                for inv_data in response_data.get("invoices", [])
            ]
            
            # Calculate totals
            total_amount = sum(inv["amount"] for inv in invoices)
            overdue_count = sum(1 for inv in invoices if Invoice(**inv).is_overdue)
            
            logger.info(
                f"Retrieved {len(invoices)} invoices for customer {customer_id}",
                extra={
                    "customer_id": customer_id,
                    "invoice_count": len(invoices),
                    "total_amount": total_amount
                }
            )
            
            return ToolResult.success_result(
                data={
                    "invoices": invoices,
                    "total_count": len(invoices),
                    "total_amount": total_amount,
                    "overdue_count": overdue_count,
                    "customer_id": customer_id
                },
                metadata={
                    "tool": self.name,
                    "operation": "get_invoices",
                    "customer_id": customer_id
                }
            )
            
        except Exception as e:
            logger.error(f"Get invoices error: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={"tool": self.name, "operation": "get_invoices"}
            )
    
    async def get_invoice_async(
        self,
        invoice_id: str
    ) -> ToolResult:
        """Get a specific invoice by ID."""
        try:
            url = f"{self.api_endpoint}/invoices/{invoice_id}"
            
            # Make API request
            response_data = await self._make_api_request(
                method="GET",
                url=url,
                operation="get_invoice"
            )
            
            # Parse invoice
            invoice = self._parse_invoice(response_data)
            
            logger.info(f"Retrieved invoice {invoice_id}")
            
            return ToolResult.success_result(
                data={
                    "invoice": invoice.to_dict(),
                    "is_overdue": invoice.is_overdue,
                    "found": True
                },
                metadata={
                    "tool": self.name,
                    "operation": "get_invoice",
                    "invoice_id": invoice_id
                }
            )
            
        except InvoiceNotFoundError:
            return ToolResult.success_result(
                data={
                    "invoice": None,
                    "found": False
                },
                metadata={
                    "tool": self.name,
                    "operation": "get_invoice",
                    "not_found": True
                }
            )
        
        except Exception as e:
            logger.error(f"Get invoice error: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={"tool": self.name, "operation": "get_invoice"}
            )
    
    async def get_balance_async(
        self,
        customer_id: str
    ) -> ToolResult:
        """Get account balance for a customer."""
        try:
            url = f"{self.api_endpoint}/customers/{customer_id}/balance"
            
            # Make API request
            response_data = await self._make_api_request(
                method="GET",
                url=url,
                operation="get_balance"
            )
            
            # Parse balance
            balance = self._parse_balance(response_data)
            
            logger.info(
                f"Retrieved balance for customer {customer_id}: {balance.current_balance}",
                extra={
                    "customer_id": customer_id,
                    "balance": balance.current_balance,
                    "outstanding": balance.total_outstanding
                }
            )
            
            return ToolResult.success_result(
                data=balance.to_dict(),
                metadata={
                    "tool": self.name,
                    "operation": "get_balance",
                    "customer_id": customer_id
                }
            )
            
        except Exception as e:
            logger.error(f"Get balance error: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={"tool": self.name, "operation": "get_balance"}
            )
    
    async def create_payment_async(
        self,
        invoice_id: str,
        amount: float,
        payment_method: str
    ) -> ToolResult:
        """Process a payment for an invoice."""
        try:
            url = f"{self.api_endpoint}/payments"
            json_data = {
                "invoice_id": invoice_id,
                "amount": amount,
                "payment_method": payment_method
            }
            
            # Make API request
            response_data = await self._make_api_request(
                method="POST",
                url=url,
                json_data=json_data,
                operation="create_payment"
            )
            
            # Parse payment
            payment = self._parse_payment(response_data)
            
            logger.info(
                f"Created payment for invoice {invoice_id}: {amount}",
                extra={
                    "invoice_id": invoice_id,
                    "payment_id": payment.payment_id,
                    "amount": amount,
                    "status": payment.status
                }
            )
            
            return ToolResult.success_result(
                data=payment.to_dict(),
                metadata={
                    "tool": self.name,
                    "operation": "create_payment",
                    "invoice_id": invoice_id,
                    "payment_id": payment.payment_id
                }
            )
            
        except PaymentFailedError as e:
            return ToolResult.error_result(
                error=f"Payment processing failed: {str(e)}",
                metadata={
                    "tool": self.name,
                    "operation": "create_payment",
                    "payment_failed": True
                }
            )
        
        except Exception as e:
            logger.error(f"Create payment error: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={"tool": self.name, "operation": "create_payment"}
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
        """Make HTTP request to Billing API."""
        if not self.session:
            raise RuntimeError("Billing tool not initialized")
        
        # Get API key securely
        api_key = tool_settings.get_billing_api_key()
        
        # Check for mock mode
        if not api_key or not self.api_endpoint:
            logger.info("Using mock billing data")
            return self._get_mock_response(method, url, operation, params, json_data)
        
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
                    raise InvoiceNotFoundError(f"Invoice not found: {url}")
                elif response.status == 402:
                    raise PaymentFailedError("Payment processing failed")
                elif response.status == 409:
                    raise InsufficientFundsError("Insufficient funds")
                elif response.status >= 400:
                    raise BillingAPIError(f"API error: {response.status}")
                
                return await response.json()
        
        return await execute_request()
    
    def _parse_invoice(self, data: Dict[str, Any]) -> Invoice:
        """Parse API response into Invoice."""
        return Invoice(
            invoice_id=data.get("id") or data.get("invoice_id"),
            customer_id=data.get("customer_id", ""),
            amount=float(data.get("amount", 0.0)),
            currency=data.get("currency", "USD"),
            status=data.get("status", "pending"),
            due_date=data.get("due_date"),
            created_at=data.get("created_at"),
            paid_at=data.get("paid_at"),
            line_items=data.get("line_items", [])
        )
    
    def _parse_payment(self, data: Dict[str, Any]) -> Payment:
        """Parse API response into Payment."""
        return Payment(
            payment_id=data.get("id") or data.get("payment_id"),
            invoice_id=data.get("invoice_id", ""),
            amount=float(data.get("amount", 0.0)),
            currency=data.get("currency", "USD"),
            status=data.get("status", "pending"),
            payment_method=data.get("payment_method", "unknown"),
            created_at=data.get("created_at")
        )
    
    def _parse_balance(self, data: Dict[str, Any]) -> AccountBalance:
        """Parse API response into AccountBalance."""
        return AccountBalance(
            customer_id=data.get("customer_id", ""),
            current_balance=float(data.get("current_balance", 0.0)),
            outstanding_invoices=int(data.get("outstanding_invoices", 0)),
            total_outstanding=float(data.get("total_outstanding", 0.0)),
            currency=data.get("currency", "USD"),
            last_payment=data.get("last_payment")
        )
    
    def _get_mock_response(
        self,
        method: str,
        url: str,
        operation: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get mock response for testing."""
        logger.info(f"Returning mock billing response for {operation}")
        
        if "get_invoices" in operation:
            return {
                "invoices": [
                    {
                        "id": "INV-2024-001",
                        "customer_id": "CUST-12345",
                        "amount": 149.99,
                        "currency": "USD",
                        "status": "paid",
                        "due_date": (datetime.utcnow() - timedelta(days=10)).isoformat() + "Z",
                        "created_at": (datetime.utcnow() - timedelta(days=40)).isoformat() + "Z",
                        "paid_at": (datetime.utcnow() - timedelta(days=35)).isoformat() + "Z",
                        "line_items": [
                            {"description": "Monthly Subscription", "amount": 149.99}
                        ]
                    },
                    {
                        "id": "INV-2024-002",
                        "customer_id": "CUST-12345",
                        "amount": 149.99,
                        "currency": "USD",
                        "status": "pending",
                        "due_date": (datetime.utcnow() + timedelta(days=5)).isoformat() + "Z",
                        "created_at": (datetime.utcnow() - timedelta(days=10)).isoformat() + "Z",
                        "line_items": [
                            {"description": "Monthly Subscription", "amount": 149.99}
                        ]
                    }
                ]
            }
        
        elif "get_invoice" in operation or ("/invoices/" in url and method == "GET"):
            return {
                "id": "INV-2024-002",
                "invoice_id": "INV-2024-002",
                "customer_id": "CUST-12345",
                "amount": 149.99,
                "currency": "USD",
                "status": "pending",
                "due_date": (datetime.utcnow() + timedelta(days=5)).isoformat() + "Z",
                "created_at": (datetime.utcnow() - timedelta(days=10)).isoformat() + "Z",
                "line_items": [
                    {"description": "Monthly Subscription", "amount": 99.99},
                    {"description": "Additional Services", "amount": 50.00}
                ]
            }
        
        elif "get_balance" in operation:
            return {
                "customer_id": "CUST-12345",
                "current_balance": -149.99,
                "outstanding_invoices": 1,
                "total_outstanding": 149.99,
                "currency": "USD",
                "last_payment": (datetime.utcnow() - timedelta(days=35)).isoformat() + "Z"
            }
        
        elif "create_payment" in operation:
            return {
                "id": "PAY-2024-001",
                "payment_id": "PAY-2024-001",
                "invoice_id": json_data.get("invoice_id") if json_data else "INV-2024-002",
                "amount": json_data.get("amount") if json_data else 149.99,
                "currency": "USD",
                "status": "completed",
                "payment_method": json_data.get("payment_method") if json_data else "credit_card",
                "created_at": datetime.utcnow().isoformat() + "Z"
            }
        
        else:
            return {"status": "ok"}


__all__ = [
    'BillingTool',
    'Invoice',
    'Payment',
    'AccountBalance',
    'BillingAPIError',
    'InvoiceNotFoundError',
    'PaymentFailedError',
    'InsufficientFundsError'
]
