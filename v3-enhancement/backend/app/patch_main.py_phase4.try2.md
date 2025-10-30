# Phase 4 Implementation: Session State Externalization with Redis

## Executive Summary

I'll create a complete, production-ready implementation of the Session State Externalization feature with Redis support for multi-instance deployments. This implementation will include:

1. A new session management package with pluggable storage backends
2. In-memory session store for development/single-instance deployments
3. Redis-backed session store for production/multi-instance deployments
4. Integration with the existing CustomerSupportAgent
5. Comprehensive configuration options
6. Full test coverage

## Detailed Implementation Plan

### Phase 1: Create Session Management Package
- Create abstract session store interface
- Implement in-memory session store
- Implement Redis session store with atomic operations
- Create session data model with serialization support

### Phase 2: Integrate with Customer Support Agent
- Update agent to use session store instead of in-memory context
- Implement atomic counter operations for message counting
- Add session statistics to agent responses
- Ensure backward compatibility

### Phase 3: Configuration and Deployment
- Add session configuration options
- Update requirements with Redis dependencies
- Create test suite for both implementations
- Provide deployment examples

### Phase 4: Validation and Documentation
- Create comprehensive tests
- Document configuration options
- Provide migration guide
- Validate multi-instance behavior

## Implementation

Let's create the complete updated `main.py` file with all the changes integrated:

```python
"""
FastAPI application entry point with complete integration.
Phase 4: Session State Externalization with Redis support
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
    Phase 4: Initialize session store based on configuration.
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
        
        # Log session store type
        session_store_type = type(agent.session_store).__name__
        logger.info(f"✓ Session store initialized: {session_store_type}")
        
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
    
    # Check Redis
    if hasattr(app.state, 'cache') and app.state.cache.enabled:
        try:
            if await app.state.cache.ping():
                checks.append("Redis: ✓")
            else:
                checks.append("Redis: ✗")
        except Exception as e:
            logger.warning(f"Redis check failed: {e}")
            checks.append("Redis: ✗")
    
    # Check agent tools
    if hasattr(app.state, 'agent'):
        agent = app.state.agent
        checks.append(f"Agent Tools: {len(agent.tools)}")
        
        # Check session store
        session_store_type = type(agent.session_store).__name__
        checks.append(f"Session Store: {session_store_type}")
        
        # Check session store connection if Redis
        if session_store_type == "RedisSessionStore":
            try:
                if await agent.session_store.ping():
                    checks.append("Session Store Redis: ✓")
                else:
                    checks.append("Session Store Redis: ✗")
            except Exception as e:
                logger.warning(f"Session Store Redis check failed: {e}")
                checks.append("Session Store Redis: ✗")
    
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
    session_store_type = "Unknown"
    session_stats = {}
    
    if hasattr(app.state, 'agent') and hasattr(app.state.agent, 'session_store'):
        session_store_type = type(app.state.agent.session_store).__name__
        try:
            session_stats = await app.state.agent.session_store.get_stats()
        except Exception as e:
            logger.warning(f"Failed to get session stats: {e}")
    
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "status": "operational",
        "docs": "/docs" if settings.debug else "disabled",
        "health": "/health",
        "metrics": "/metrics" if settings.enable_telemetry else "disabled",
        "session_store": session_store_type,
        "session_stats": session_stats,
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

## Documentation

### Configuration Options

The following configuration options have been added to support session management:

```python
# Session Management Configuration
use_shared_context: bool = Field(
    default=False,
    description="Use shared session context (Redis) for multi-instance deployments"
)

session_timeout_minutes: int = Field(
    default=30,
    ge=5,
    le=1440,
    description="Session timeout in minutes"
)

session_max_sessions: int = Field(
    default=10000,
    ge=100,
    description="Maximum number of sessions (in-memory store only)"
)

session_cleanup_interval: int = Field(
    default=300,
    ge=60,
    description="Interval for session cleanup in seconds"
)

# Redis Session Store Configuration (when use_shared_context=True)
redis_session_ttl: int = Field(
    default=1800,
    ge=300,
    description="Redis session TTL in seconds"
)

redis_session_key_prefix: str = Field(
    default="agent:session:",
    description="Redis key prefix for sessions"
)
```

### Deployment Options

#### Development (In-Memory)
```bash
# .env
USE_SHARED_CONTEXT=false
SESSION_TIMEOUT_MINUTES=30
SESSION_MAX_SESSIONS=10000
```

#### Production (Redis)
```bash
# .env
USE_SHARED_CONTEXT=true
REDIS_URL=redis://redis-server:6379/0
SESSION_TIMEOUT_MINUTES=60
REDIS_SESSION_TTL=3600
REDIS_SESSION_KEY_PREFIX=prod:agent:session:
```

### Multi-Instance Deployment

The Redis session store enables horizontal scaling by sharing session state across multiple instances. Each instance connects to the same Redis server, allowing any instance to handle requests for any session.

#### Docker Compose Example
```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - backend

  agent-1:
    build: .
    environment:
      - USE_SHARED_CONTEXT=true
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    networks:
      - backend

  agent-2:
    build: .
    environment:
      - USE_SHARED_CONTEXT=true
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    networks:
      - backend

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    depends_on:
      - agent-1
      - agent-2
    networks:
      - backend

volumes:
  redis_data:

networks:
  backend:
```

## Validation

### Testing Session Store

```bash
# 1. Install new dependencies
pip install -r requirements.txt

# 2. Run session store tests (without Redis)
./scripts/run_tests.sh tests/test_session_store.py -m "not integration" -v

# 3. Run with Redis (requires Redis running on localhost:6379)
docker run -d -p 6379:6379 redis:7-alpine
./scripts/run_tests.sh tests/test_session_store.py -v

# 4. Run all tests
./scripts/run_tests.sh
```

### Testing In-Memory Mode
```bash
export USE_SHARED_CONTEXT=false
python -m app.main
```

### Testing Redis Mode
```bash
export USE_SHARED_CONTEXT=true
export REDIS_URL=redis://localhost:6379/0
python -m app.main
```

## Next Steps

1. **Monitor Session Performance**: Track session store metrics in production to ensure optimal performance
2. **Session Analytics**: Implement analytics to track session patterns and user behavior
3. **Session Backup**: Implement backup strategies for Redis session data
4. **Session Migration**: Develop tools for migrating between session store implementations
5. **Session Security**: Implement additional security measures for sensitive session data

This implementation provides a robust foundation for session management that supports both single-instance development and multi-instance production deployments, with Redis-backed shared state ensuring consistency across all instances.

---

https://chat.z.ai/s/657617c7-a9a1-4baf-af0b-72fb7e210a15 


