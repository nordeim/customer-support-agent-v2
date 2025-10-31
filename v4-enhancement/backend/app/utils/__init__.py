"""
Utility modules for the application.
Provides encryption, retry logic, telemetry, and middleware.

Version: 1.0.0
"""

# Encryption utilities
try:
    from .encryption import (
        SessionEncryption,
        TimestampedEncryption,
        EncryptionError,
        create_encryption_instance
    )
    ENCRYPTION_AVAILABLE = True
except ImportError:
    SessionEncryption = None
    TimestampedEncryption = None
    EncryptionError = None
    create_encryption_instance = None
    ENCRYPTION_AVAILABLE = False

# Retry utilities
try:
    from .retry import (
        RetryConfig,
        RetryStrategy,
        CircuitBreaker,
        CircuitBreakerState,
        CircuitBreakerOpenError,
        retry,
        async_retry,
        calculate_retry_delay
    )
    RETRY_AVAILABLE = True
except ImportError:
    RetryConfig = None
    RetryStrategy = None
    CircuitBreaker = None
    CircuitBreakerState = None
    CircuitBreakerOpenError = None
    retry = None
    async_retry = None
    calculate_retry_delay = None
    RETRY_AVAILABLE = False

# Telemetry utilities
try:
    from .telemetry import (
        setup_telemetry,
        metrics_collector,
        get_tracer,
        trace_async,
        trace_sync
    )
    TELEMETRY_AVAILABLE = True
except ImportError:
    setup_telemetry = None
    metrics_collector = None
    get_tracer = None
    trace_async = None
    trace_sync = None
    TELEMETRY_AVAILABLE = False

# Middleware
try:
    from .middleware import (
        RequestIDMiddleware,
        TimingMiddleware,
        RateLimitMiddleware,
        ErrorHandlingMiddleware
    )
    MIDDLEWARE_AVAILABLE = True
except ImportError:
    RequestIDMiddleware = None
    TimingMiddleware = None
    RateLimitMiddleware = None
    ErrorHandlingMiddleware = None
    MIDDLEWARE_AVAILABLE = False


__all__ = [
    # Encryption
    'SessionEncryption',
    'TimestampedEncryption',
    'EncryptionError',
    'create_encryption_instance',
    
    # Retry
    'RetryConfig',
    'RetryStrategy',
    'CircuitBreaker',
    'CircuitBreakerState',
    'CircuitBreakerOpenError',
    'retry',
    'async_retry',
    'calculate_retry_delay',
    
    # Telemetry
    'setup_telemetry',
    'metrics_collector',
    'get_tracer',
    'trace_async',
    'trace_sync',
    
    # Middleware
    'RequestIDMiddleware',
    'TimingMiddleware',
    'RateLimitMiddleware',
    'ErrorHandlingMiddleware',
    
    # Availability flags
    'ENCRYPTION_AVAILABLE',
    'RETRY_AVAILABLE',
    'TELEMETRY_AVAILABLE',
    'MIDDLEWARE_AVAILABLE'
]
