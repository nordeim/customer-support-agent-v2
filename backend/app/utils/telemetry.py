"""
Telemetry and monitoring utilities.
"""
import logging
from typing import Any
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import FastAPI, Response
import time

logger = logging.getLogger(__name__)

# Metrics definitions
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

chat_messages = Counter(
    'chat_messages_total',
    'Total chat messages',
    ['session_id', 'role', 'escalated']
)

active_sessions = Gauge(
    'active_sessions_count',
    'Number of active chat sessions'
)

tool_usage = Counter(
    'tool_usage_total',
    'Tool usage statistics',
    ['tool_name', 'success']
)

websocket_connections = Gauge(
    'websocket_connections_active',
    'Active WebSocket connections'
)

# Custom metrics for business logic
escalations = Counter(
    'escalations_total',
    'Total escalations to human support',
    ['reason', 'priority']
)

response_time = Histogram(
    'agent_response_time_seconds',
    'Agent response generation time',
    ['with_rag', 'with_escalation']
)

cache_operations = Counter(
    'cache_operations_total',
    'Cache operations',
    ['operation', 'hit']
)


def setup_telemetry(app: FastAPI) -> None:
    """
    Setup telemetry and monitoring for the application.
    
    Args:
        app: FastAPI application instance
    """
    logger.info("Setting up telemetry...")
    
    # Add metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(content=generate_latest(), media_type="text/plain")
    
    # Add middleware for request metrics
    @app.middleware("http")
    async def track_requests(request, call_next):
        """Track HTTP request metrics."""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        
        request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        request_duration.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        return response
    
    logger.info("Telemetry setup complete")


def track_chat_message(session_id: str, role: str, escalated: bool = False) -> None:
    """Track chat message metrics."""
    chat_messages.labels(
        session_id=session_id,
        role=role,
        escalated=str(escalated)
    ).inc()


def track_tool_usage(tool_name: str, success: bool = True) -> None:
    """Track tool usage metrics."""
    tool_usage.labels(
        tool_name=tool_name,
        success=str(success)
    ).inc()


def track_escalation(reason: str, priority: str = "normal") -> None:
    """Track escalation metrics."""
    escalations.labels(
        reason=reason,
        priority=priority
    ).inc()


def track_response_time(duration: float, with_rag: bool = False, with_escalation: bool = False) -> None:
    """Track agent response time."""
    response_time.labels(
        with_rag=str(with_rag),
        with_escalation=str(with_escalation)
    ).observe(duration)


def track_cache_operation(operation: str, hit: bool = False) -> None:
    """Track cache operations."""
    cache_operations.labels(
        operation=operation,
        hit=str(hit)
    ).inc()


def update_active_sessions(count: int) -> None:
    """Update active sessions gauge."""
    active_sessions.set(count)


def update_websocket_connections(count: int) -> None:
    """Update WebSocket connections gauge."""
    websocket_connections.set(count)


class MetricsCollector:
    """Collects and manages application metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.message_count = 0
        self.error_count = 0
    
    def record_message(self, session_id: str, role: str, escalated: bool = False):
        """Record a chat message."""
        self.message_count += 1
        track_chat_message(session_id, role, escalated)
    
    def record_error(self):
        """Record an error."""
        self.error_count += 1
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        uptime = time.time() - self.start_time
        
        return {
            "uptime_seconds": uptime,
            "messages_processed": self.message_count,
            "errors": self.error_count,
            "messages_per_minute": (self.message_count / uptime) * 60 if uptime > 0 else 0
        }


# Global metrics collector
metrics_collector = MetricsCollector()
