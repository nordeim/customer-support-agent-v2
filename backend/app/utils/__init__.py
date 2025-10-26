"""
Utilities module for Customer Support AI Agent
"""

from .middleware import (
    RequestIDMiddleware,
    TimingMiddleware,
    RateLimitMiddleware,
    ErrorHandlingMiddleware
)

from .telemetry import (
    setup_telemetry,
    track_chat_message,
    track_tool_usage,
    track_escalation,
    track_response_time,
    track_cache_operation,
    update_active_sessions,
    update_websocket_connections,
    metrics_collector
)

__all__ = [
    # Middleware
    "RequestIDMiddleware",
    "TimingMiddleware", 
    "RateLimitMiddleware",
    "ErrorHandlingMiddleware",
    
    # Telemetry
    "setup_telemetry",
    "track_chat_message",
    "track_tool_usage",
    "track_escalation",
    "track_response_time",
    "track_cache_operation",
    "update_active_sessions",
    "update_websocket_connections",
    "metrics_collector",
]
