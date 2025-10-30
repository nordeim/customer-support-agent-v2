# Phase 1: Request Analysis & Planning

## Executive Summary
I will now review all three files together - the original `main.py`, the abstract `session_store.py`, and the concrete `redis_session_store.py` - to understand the complete session store architecture and provide the correct integration for the requested modification.

## Deep Analysis & Deliberation

### Complete Architecture Analysis

**File 1: `session_store.py` (Abstract Interface)**
- Defines `SessionData` dataclass with serialization methods
- Defines abstract `SessionStore` base class with all required methods
- Establishes the contract that all session stores must follow

**File 2: `redis_session_store.py` (Concrete Implementation)**
- Implements `SessionStore` interface using Redis
- Requires Redis connection parameters (redis_url, key_prefix, default_ttl, max_connections)
- Has async initialization pattern with `_ensure_connection()`
- Provides `ping()` method for health checks
- Has `close()` method for cleanup
- Uses connection pooling and Lua scripts for atomic operations

**File 3: Original `main.py` (Base Application)**
- Has comprehensive lifecycle management
- Initializes database, cache, telemetry, and agent
- Has pattern for service initialization with health checks
- Includes cleanup in shutdown section

### Original Request Analysis

The original request was to update the `lifespan` function to initialize session store. Looking at the provided change, it appears to be a simplified version that removes some existing functionality rather than adding session store.

### Correct Integration Requirements

Based on the concrete `RedisSessionStore` implementation, I need to:

1. **Import RedisSessionStore**: From the correct module path
2. **Initialize with Parameters**: RedisSessionStore requires specific parameters
3. **Follow Existing Patterns**: Use the same initialization pattern as cache service
4. **Handle Redis Dependencies**: Check if Redis is available
5. **Add to Health Checks**: Include session store in startup checks
6. **Proper Cleanup**: Call close() method during shutdown

### Risk Assessment

- **Medium Risk**: RedisSessionStore has specific initialization requirements
- **Low Risk**: Follows same patterns as other services
- **Low Risk**: Has proper error handling and health check methods

## Execution Plan

### Phase 1: Determine Correct Import Path
Based on the file structure, the import should be:
```python
from .session.redis_session_store import RedisSessionStore
```

### Phase 2: Implement Session Store Initialization
1. Add RedisSessionStore import
2. Initialize with Redis URL from settings
3. Test connection with ping()
4. Add to app.state
5. Include in startup health checks
6. Add cleanup in shutdown

### Phase 3: Handle Configuration
1. Add Redis session store settings to configuration
2. Use sensible defaults for missing settings
3. Handle Redis unavailability gracefully

# Phase 2: Implementation

Based on my analysis of all three files, here's the correct integration of session store into `main.py`:

```python
"""
FastAPI application entry point with complete integration.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import asyncio
from typing import Any, Dict

from .config import settings
from .api.routes import chat, sessions, health
from .api.websocket import websocket_endpoint
from .agents.chat_agent import CustomerSupportAgent
from .utils.telemetry import setup_telemetry, metrics_collector
from .utils.middleware import (
    RequestIDMiddleware, 
    TimingMiddleware, 
    RateLimitMiddleware,
    ErrorHandlingMiddleware
)
from .database import init_db, cleanup_db
from .services.cache_service import CacheService
from .session.redis_session_store import RedisSessionStore, REDIS_AVAILABLE

# Configure structured logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/app.log', mode='a') if settings.environment != 'development' else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle events.
    Initialize resources on startup, cleanup on shutdown.
    """
    # Startup
    try:
        logger.info("=" * 60)
        logger.info(f"Starting {settings.app_name} v{settings.app_version}")
        logger.info(f"Environment: {settings.environment}")
        logger.info("=" * 60)
        
        # Initialize database
        logger.info("Initializing database...")
        init_db()
        
        # Initialize cache service
        logger.info("Initializing cache service...")
        cache_service = CacheService()
        app.state.cache = cache_service
        
        # Test cache connection with better error handling
        try:
            if await cache_service.ping():
                logger.info("✓ Cache service connected")
            else:
                logger.warning("✗ Cache service unavailable - running without cache")
        except Exception as e:
            logger.warning(f"✗ Cache service unavailable - running without cache: {e}")
        
        # Initialize session store
        session_store = None
        if REDIS_AVAILABLE:
            logger.info("Initializing session store...")
            try:
                # Get Redis URL from settings or use default
                redis_url = getattr(settings, 'redis_session_url', None)
                if not redis_url:
                    # Try to construct from individual settings
                    redis_host = getattr(settings, 'redis_host', 'localhost')
                    redis_port = getattr(settings, 'redis_port', 6379)
                    redis_db = getattr(settings, 'redis_session_db', 1)
                    redis_password = getattr(settings, 'redis_password', None)
                    
                    if redis_password:
                        redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}"
                    else:
                        redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
                
                # Initialize Redis session store
                session_store = RedisSessionStore(
                    redis_url=redis_url,
                    key_prefix=getattr(settings, 'session_key_prefix', 'session:'),
                    default_ttl=getattr(settings, 'session_ttl', 3600),
                    max_connections=getattr(settings, 'session_max_connections', 10)
                )
                
                app.state.session_store = session_store
                
                # Test session store connection
                if await session_store.ping():
                    logger.info("✓ Session store connected")
                else:
                    logger.warning("✗ Session store unavailable - running without session persistence")
                    app.state.session_store = None
                    session_store = None
                    
            except Exception as e:
                logger.error(f"Failed to initialize session store: {e}")
                session_store = None
                app.state.session_store = None
        else:
            logger.warning("Redis not available for session store - running without session persistence")
        
        # Initialize telemetry
        if settings.enable_telemetry:
            setup_telemetry(app)
            logger.info("✓ Telemetry initialized")
        
        # Initialize the AI agent
        logger.info("Initializing AI agent...")
        agent = CustomerSupportAgent()

        # If using registry mode, initialize asynchronously
        if agent.use_registry:
            await agent.initialize_async()
        else:
            # Legacy mode already initialized in __init__
            pass
        
        app.state.agent = agent
        logger.info("✓ AI agent initialized successfully")
        
        # Add sample data to knowledge base
        if settings.environment == "development":
            await add_sample_knowledge(agent)
        
        # Perform startup health checks
        await perform_startup_checks(app)
        
        logger.info("=" * 60)
        logger.info("Application started successfully")
        logger.info(f"API docs available at: http://{settings.api_host}:{settings.api_port}/docs")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("=" * 60)
    logger.info("Shutting down application...")
    
    try:
        # Cleanup agent resources
        if hasattr(app.state, 'agent'):
            await app.state.agent.cleanup()
            logger.info("✓ Agent cleanup complete")
        
        # Close session store connections
        if hasattr(app.state, 'session_store') and app.state.session_store is not None:
            try:
                await app.state.session_store.close()
                logger.info("✓ Session store connections closed")
            except Exception as e:
                logger.warning(f"Error closing session store: {e}")
        
        # Close cache connections
        if hasattr(app.state, 'cache'):
            await app.state.cache.close()
            logger.info("✓ Cache connections closed")
        
        # Cleanup database
        cleanup_db()
        logger.info("✓ Database cleanup complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    logger.info("Application shutdown complete")
    logger.info("=" * 60)


async def perform_startup_checks(app: FastAPI) -> None:
    """Perform critical health checks on startup."""
    checks = []
    
    # Check database connection
    try:
        from .database import check_db_connection, check_tables_exist
        if check_db_connection():
            checks.append("Database: ✓")
            
            # Check if tables exist
            if check_tables_exist():
                checks.append("Tables: ✓")
            else:
                checks.append("Tables: ✗")
                logger.error("Required database tables are missing")
        else:
            checks.append("Database: ✗")
            logger.error("Database connection failed")
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        checks.append("Database: ✗")
    
    # Check Redis (cache)
    if hasattr(app.state, 'cache') and app.state.cache.enabled:
        try:
            if await app.state.cache.ping():
                checks.append("Redis Cache: ✓")
            else:
                checks.append("Redis Cache: ✗")
        except Exception as e:
            logger.warning(f"Redis cache check failed: {e}")
            checks.append("Redis Cache: ✗")
    
    # Check session store
    if hasattr(app.state, 'session_store') and app.state.session_store is not None:
        try:
            if await app.state.session_store.ping():
                checks.append("Session Store: ✓")
            else:
                checks.append("Session Store: ✗")
        except Exception as e:
            logger.warning(f"Session store check failed: {e}")
            checks.append("Session Store: ✗")
    else:
        checks.append("Session Store: ⚠ (disabled)")
    
    # Check agent tools
    if hasattr(app.state, 'agent'):
        agent = app.state.agent
        checks.append(f"Agent Tools: {len(agent.tools)}")
    
    logger.info(f"Startup checks: {', '.join(checks)}")
    
    # Fail startup if critical components are not ready
    if "Database: ✗" in checks or "Tables: ✗" in checks:
        logger.error("Critical startup checks failed")
        raise RuntimeError("Application startup failed due to critical component issues")


async def add_sample_knowledge(agent: CustomerSupportAgent) -> None:
    """Add sample documents to knowledge base for development."""
    try:
        rag_tool = agent.tools.get('rag')
        if not rag_tool:
            return
        
        sample_docs = [
            "Welcome to our customer support! We're available 24/7 to help you.",
            "To reset your password: 1. Click 'Forgot Password' 2. Enter your email 3. Check your inbox 4. Follow the reset link.",
            "Our refund policy: Full refunds are available within 30 days of purchase for unused items in original condition.",
            "Shipping information: Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.",
            "Account verification requires: Valid email address, phone number, and government-issued ID for certain features.",
            "Technical support hours: Available 24/7 via chat. Phone support available Mon-Fri 9AM-6PM EST.",
            "Premium membership benefits: Free shipping, priority support, exclusive discounts, early access to sales.",
            "Payment methods accepted: Credit cards (Visa, MasterCard, Amex), PayPal, Apple Pay, Google Pay.",
            "Order tracking: Use your order number on our tracking page or contact support for assistance.",
            "Data privacy: We encrypt all personal data and never share information with third parties without consent."
        ]
        
        result = rag_tool.add_documents(sample_docs)
        logger.info(f"Added {result.get('documents_added', 0)} sample documents to knowledge base")
        
    except Exception as e:
        logger.warning(f"Failed to add sample knowledge: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered customer support system with RAG, memory, and intelligent escalation",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time", "X-RateLimit-Limit"]
)

# Add custom middleware (order matters - applied in reverse)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(TimingMiddleware)
if settings.rate_limit_enabled:
    app.add_middleware(
        RateLimitMiddleware,
        calls=settings.rate_limit_requests,
        period=settings.rate_limit_period
    )

# Include API routes
app.include_router(
    health.router,
    prefix="/health",
    tags=["Health"]
)

app.include_router(
    sessions.router,
    prefix=f"{settings.api_prefix}/sessions",
    tags=["Sessions"]
)

app.include_router(
    chat.router,
    prefix=f"{settings.api_prefix}/chat",
    tags=["Chat"]
)

# Add WebSocket endpoint
app.add_api_websocket_route(
    "/ws",
    websocket_endpoint,
    name="websocket"
)

# Root endpoint
@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """
    Root endpoint with API information.
    
    Returns:
        API information and status
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "status": "operational",
        "docs": "/docs" if settings.debug else "disabled",
        "health": "/health",
        "metrics": "/metrics" if settings.enable_telemetry else "disabled",
        "stats": metrics_collector.get_stats()
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions gracefully."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Special handling for database errors
    if "no such table" in str(exc):
        logger.error(
            f"Database table error in request {request_id}: {exc}",
            exc_info=True,
            extra={
                "path": request.url.path,
                "method": request.method,
                "client": request.client.host if request.client else "unknown"
            }
        )
        
        # Try to recover by checking database state
        try:
            from .database import check_tables_exist
            if not check_tables_exist():
                logger.error("Database tables are missing - this indicates an initialization issue")
        except Exception as db_check_error:
            logger.error(f"Failed to check database state: {db_check_error}")
        
        metrics_collector.record_error()
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Database configuration error",
                "message": "The application database is not properly initialized",
                "request_id": request_id,
                "support": "Please contact support if this persists"
            }
        )
    
    # Standard error handling for other exceptions
    logger.error(
        f"Unhandled exception in request {request_id}: {exc}",
        exc_info=True,
        extra={
            "path": request.url.path,
            "method": request.method,
            "client": request.client.host if request.client else "unknown"
        }
    )
    
    metrics_collector.request_id = getattr(request.state, "request_id", "unknown")
    metrics_collector.record_error()
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.debug else "An unexpected error occurred",
            "request_id": request_id,
            "support": "Please contact support if this persists"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Create logs directory if it doesn't exist
    import os
    os.makedirs("logs", exist_ok=True)
    
    # Run with uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
        access_log=True,
        use_colors=True
    )
```

# Phase 3: Validation & Refinement

## Key Integration Points

1. **Correct Import Path**: 
   ```python
   from .session.redis_session_store import RedisSessionStore, REDIS_AVAILABLE
   ```

2. **Proper Initialization**:
   - Checks REDIS_AVAILABLE before attempting initialization
   - Constructs Redis URL from individual settings if full URL not provided
   - Passes all required parameters to RedisSessionStore constructor

3. **Configuration Handling**:
   - Uses settings for Redis connection parameters
   - Provides sensible defaults for missing settings
   - Supports both full URL and individual connection parameters

4. **Error Handling**:
   - Gracefully handles Redis unavailability
   - Sets session_store to None if initialization fails
   - Continues application startup even if session store fails

5. **Health Checks**:
   - Includes session store in startup checks
   - Shows "disabled" status when session store is not available
   - Distinguishes between Redis cache and session store in checks

6. **Cleanup**:
   - Properly calls close() method during shutdown
   - Handles cleanup errors gracefully

## Quality Assurance Verification

- [x] Correctly imports RedisSessionStore from the right module
- [x] Handles Redis availability check
- [x] Initializes with proper parameters
- [x] Follows existing patterns for service initialization
- [x] Includes comprehensive error handling
- [x] Adds session store to health checks
- [x] Implements proper cleanup
- [x] Maintains backward compatibility

# Phase 4: Delivery & Knowledge Transfer

## Complete Solution

The updated `main.py` now correctly integrates the Redis session store:

1. **Imports**: Correctly imports `RedisSessionStore` and `REDIS_AVAILABLE` flag
2. **Initialization**: Initializes session store with proper Redis configuration
3. **Health Checks**: Includes session store in startup health checks
4. **Error Handling**: Gracefully handles session store unavailability
5. **Cleanup**: Properly closes session store connections during shutdown

## Configuration Requirements

The session store expects these settings (with defaults):
- `redis_session_url` or individual Redis connection settings
- `session_key_prefix` (default: 'session:')
- `session_ttl` (default: 3600 seconds)
- `session_max_connections` (default: 10)

## Usage

The session store is now available via `app.state.session_store` and can be used in other parts of the application for session persistence. The application will continue to run even if Redis is unavailable, but without session persistence.

This implementation follows all established patterns in the codebase and provides a robust, production-ready integration of the Redis session store.

---

https://chat.z.ai/s/da1d5ba3-87e4-4f16-bd61-55d0928bc7cf 

