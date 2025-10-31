"""
FastAPI application entry point with complete integration.
Version: 2.0.0 (Enhanced session management, health monitoring, graceful shutdown)
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import asyncio
from typing import Any, Dict
import signal
import sys

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
from .database import init_db, cleanup_db, check_db_connection, check_tables_exist
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

# Global shutdown flag
shutdown_event = asyncio.Event()


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


async def periodic_cleanup_task(app: FastAPI) -> None:
    """
    Background task for periodic cleanup.
    Runs session cleanup and health checks.
    """
    logger.info("Starting periodic cleanup task")
    
    cleanup_interval = settings.session_cleanup_interval_seconds
    
    while not shutdown_event.is_set():
        try:
            # Wait for cleanup interval or shutdown
            await asyncio.wait_for(
                shutdown_event.wait(),
                timeout=cleanup_interval
            )
            # If we get here, shutdown was signaled
            break
            
        except asyncio.TimeoutError:
            # Timeout means it's time for cleanup
            pass
        
        try:
            # Cleanup expired sessions
            if hasattr(app.state, 'agent'):
                agent = app.state.agent
                cleaned = await agent.session_store.cleanup_expired()
                
                if cleaned > 0:
                    logger.info(f"Periodic cleanup: removed {cleaned} expired sessions")
                
                # Get session stats
                stats = await agent.session_store.get_stats()
                logger.debug(f"Session store stats: {stats}")
            
            # Health check on Redis (if using Redis session store)
            if hasattr(app.state, 'agent'):
                from .session import RedisSessionStore
                if isinstance(app.state.agent.session_store, RedisSessionStore):
                    redis_healthy = await app.state.agent.session_store.ping()
                    if not redis_healthy:
                        logger.warning("Redis session store health check failed")
            
            # Cache health check
            if hasattr(app.state, 'cache') and app.state.cache.enabled:
                cache_healthy = await app.state.cache.ping()
                if not cache_healthy:
                    logger.warning("Cache service health check failed")
            
        except Exception as e:
            logger.error(f"Error in periodic cleanup task: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle events.
    Initialize resources on startup, cleanup on shutdown.
    Version 2.0.0: Enhanced with health monitoring and graceful shutdown.
    """
    # === STARTUP ===
    startup_tasks = []
    cleanup_task = None
    
    try:
        logger.info("=" * 60)
        logger.info(f"Starting {settings.app_name} v{settings.app_version}")
        logger.info(f"Environment: {settings.environment}")
        logger.info(f"Debug mode: {settings.debug}")
        logger.info("=" * 60)
        
        # Create logs directory
        import os
        os.makedirs("logs", exist_ok=True)
        
        # Initialize database
        logger.info("Initializing database...")
        init_db()
        
        # Verify database
        if not check_db_connection():
            raise RuntimeError("Database connection check failed")
        
        if not check_tables_exist():
            logger.warning("Some database tables are missing, attempting to create...")
            init_db()
            
            if not check_tables_exist():
                raise RuntimeError("Failed to create required database tables")
        
        logger.info("✓ Database initialized and verified")
        
        # Initialize cache service
        logger.info("Initializing cache service...")
        cache_service = CacheService()
        app.state.cache = cache_service
        
        # Test cache connection
        try:
            if await cache_service.ping():
                logger.info("✓ Cache service connected")
            else:
                logger.warning("✗ Cache service unavailable - running without cache")
        except Exception as e:
            logger.warning(f"✗ Cache service unavailable: {e}")
        
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
        
        app.state.agent = agent
        logger.info("✓ AI agent initialized successfully")
        
        # Log session store configuration
        session_store_type = type(agent.session_store).__name__
        logger.info(f"✓ Session store: {session_store_type}")
        
        if hasattr(agent.session_store, 'enable_l1_cache'):
            logger.info(f"  - L1 cache: {agent.session_store.enable_l1_cache}")
        
        if hasattr(agent.session_store, 'enable_encryption'):
            logger.info(f"  - Encryption: {agent.session_store.enable_encryption}")
        
        if agent.lock_manager:
            logger.info(f"  - Distributed locking: enabled")
        
        # Test session store
        try:
            session_health = await agent.session_store.health_check()
            if session_health.get('healthy'):
                logger.info("✓ Session store health check passed")
            else:
                logger.warning(f"✗ Session store health check failed: {session_health}")
        except Exception as e:
            logger.warning(f"✗ Session store health check error: {e}")
        
        # Add sample data in development
        if settings.environment == "development" and settings.dev_sample_data:
            await add_sample_knowledge(agent)
        
        # Perform comprehensive startup checks
        await perform_startup_checks(app)
        
        # Start periodic cleanup task
        logger.info("Starting background cleanup task...")
        cleanup_task = asyncio.create_task(periodic_cleanup_task(app))
        startup_tasks.append(cleanup_task)
        
        logger.info("=" * 60)
        logger.info("✓ Application started successfully")
        logger.info(f"API docs: http://{settings.api_host}:{settings.api_port}/docs")
        logger.info(f"Health check: http://{settings.api_host}:{settings.api_port}/health")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        raise
    
    yield  # === APPLICATION RUNS HERE ===
    
    # === SHUTDOWN ===
    logger.info("=" * 60)
    logger.info("Shutting down application...")
    logger.info("=" * 60)
    
    # Signal shutdown to background tasks
    shutdown_event.set()
    
    # Wait for cleanup task to finish (with timeout)
    if cleanup_task and not cleanup_task.done():
        logger.info("Waiting for cleanup task to complete...")
        try:
            await asyncio.wait_for(cleanup_task, timeout=5.0)
            logger.info("✓ Cleanup task completed")
        except asyncio.TimeoutError:
            logger.warning("Cleanup task did not complete in time, cancelling...")
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                logger.info("✓ Cleanup task cancelled")
    
    # Cleanup agent resources
    if hasattr(app.state, 'agent'):
        try:
            logger.info("Cleaning up agent resources...")
            await app.state.agent.cleanup()
            logger.info("✓ Agent cleanup complete")
        except Exception as e:
            logger.error(f"Error during agent cleanup: {e}")
    
    # Close cache connections
    if hasattr(app.state, 'cache'):
        try:
            logger.info("Closing cache connections...")
            await app.state.cache.close()
            logger.info("✓ Cache connections closed")
        except Exception as e:
            logger.error(f"Error closing cache: {e}")
    
    # Cleanup database
    try:
        logger.info("Cleaning up database connections...")
        cleanup_db()
        logger.info("✓ Database cleanup complete")
    except Exception as e:
        logger.error(f"Error during database cleanup: {e}")
    
    logger.info("=" * 60)
    logger.info("✓ Application shutdown complete")
    logger.info("=" * 60)


async def perform_startup_checks(app: FastAPI) -> None:
    """
    Perform critical health checks on startup.
    
    Args:
        app: FastAPI application instance
        
    Raises:
        RuntimeError: If critical components are not ready
    """
    checks = []
    critical_failures = []
    
    # Database check
    try:
        if check_db_connection():
            checks.append("Database: ✓")
            
            if check_tables_exist():
                checks.append("Tables: ✓")
            else:
                checks.append("Tables: ✗")
                critical_failures.append("Database tables missing")
        else:
            checks.append("Database: ✗")
            critical_failures.append("Database connection failed")
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        checks.append("Database: ✗")
        critical_failures.append(f"Database error: {e}")
    
    # Redis/Cache check
    if hasattr(app.state, 'cache') and app.state.cache.enabled:
        try:
            if await app.state.cache.ping():
                checks.append("Redis Cache: ✓")
            else:
                checks.append("Redis Cache: ✗")
                # Not critical if cache is unavailable
        except Exception as e:
            logger.warning(f"Redis cache check failed: {e}")
            checks.append("Redis Cache: ✗")
    
    # Agent tools check
    if hasattr(app.state, 'agent'):
        agent = app.state.agent
        checks.append(f"Agent Tools: {len(agent.tools)}")
        
        # Session store check
        session_store_type = type(agent.session_store).__name__
        checks.append(f"Session Store: {session_store_type}")
        
        # Session store health check
        try:
            health = await agent.session_store.health_check()
            if health.get('healthy'):
                checks.append("Session Store: ✓")
            else:
                checks.append("Session Store: ✗")
                logger.warning(f"Session store health: {health}")
        except Exception as e:
            checks.append("Session Store: ✗")
            logger.warning(f"Session store health check failed: {e}")
        
        # Redis session store specific check
        if session_store_type == "RedisSessionStore":
            try:
                if await agent.session_store.ping():
                    checks.append("Session Store Redis: ✓")
                else:
                    checks.append("Session Store Redis: ✗")
                    critical_failures.append("Redis session store unavailable")
            except Exception as e:
                logger.warning(f"Session store Redis check failed: {e}")
                checks.append("Session Store Redis: ✗")
                critical_failures.append(f"Redis session store error: {e}")
        
        # Distributed lock check
        if agent.lock_manager:
            checks.append("Distributed Locking: ✓")
    
    # Log all checks
    logger.info("Startup health checks:")
    for check in checks:
        logger.info(f"  {check}")
    
    # Fail startup if critical components are not ready
    if critical_failures:
        logger.error("CRITICAL STARTUP FAILURES:")
        for failure in critical_failures:
            logger.error(f"  - {failure}")
        
        raise RuntimeError(
            f"Application startup failed due to critical issues: {', '.join(critical_failures)}"
        )


async def add_sample_knowledge(agent: CustomerSupportAgent) -> None:
    """
    Add sample documents to knowledge base for development.
    
    Args:
        agent: CustomerSupportAgent instance
    """
    try:
        rag_tool = agent.tools.get('rag')
        if not rag_tool:
            logger.debug("RAG tool not available, skipping sample data")
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
            "Data privacy: We encrypt all personal data and never share information with third parties without consent.",
            "International shipping: We ship to over 100 countries. Customs fees may apply.",
            "Product warranty: All products come with a 1-year manufacturer warranty. Extended warranties available.",
            "Return process: Contact support for a return authorization number, then ship items back within 30 days.",
            "Live chat hours: 24/7 automated support, human agents available Mon-Fri 8AM-10PM EST.",
            "Account security: Enable two-factor authentication for enhanced security. Change password every 90 days."
        ]
        
        result = rag_tool.add_documents(sample_docs)
        logger.info(f"✓ Added {result.get('documents_added', 0)} sample documents to knowledge base")
        
    except Exception as e:
        logger.warning(f"Failed to add sample knowledge: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered customer support system with RAG, memory, intelligent escalation, and distributed session management",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
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
    Root endpoint with API information and health status.
    
    Returns:
        API information, version, and system status
    """
    session_store_type = "Unknown"
    session_stats = {}
    session_health = {}
    
    if hasattr(app.state, 'agent') and hasattr(app.state.agent, 'session_store'):
        session_store_type = type(app.state.agent.session_store).__name__
        
        try:
            session_stats = await app.state.agent.session_store.get_stats()
        except Exception as e:
            logger.warning(f"Failed to get session stats: {e}")
        
        try:
            session_health = await app.state.agent.session_store.health_check()
        except Exception as e:
            logger.warning(f"Failed to get session health: {e}")
    
    # Get database info
    from .database import get_database_info
    db_info = get_database_info()
    
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "status": "operational",
        "endpoints": {
            "docs": "/docs" if settings.debug else "disabled",
            "health": "/health",
            "metrics": "/metrics" if settings.enable_telemetry else "disabled",
            "api": settings.api_prefix
        },
        "session_management": {
            "store_type": session_store_type,
            "stats": session_stats,
            "health": session_health
        },
        "database": db_info,
        "features": {
            "distributed_locking": hasattr(app.state.agent, 'lock_manager') and app.state.agent.lock_manager is not None if hasattr(app.state, 'agent') else False,
            "session_encryption": settings.session_encryption_enabled,
            "session_fingerprinting": settings.session_fingerprinting_enabled,
            "l1_cache": settings.session_l1_cache_enabled,
            "telemetry": settings.enable_telemetry
        },
        "metrics": metrics_collector.get_stats()
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle uncaught exceptions gracefully.
    
    Args:
        request: FastAPI request
        exc: Exception that was raised
        
    Returns:
        JSON error response
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Special handling for database errors
    if "no such table" in str(exc).lower():
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
                logger.error("Database tables are missing - initialization issue detected")
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
    
    # Security error handling
    if "SecurityError" in type(exc).__name__:
        logger.warning(
            f"Security error in request {request_id}: {exc}",
            extra={
                "path": request.url.path,
                "method": request.method,
                "client": request.client.host if request.client else "unknown"
            }
        )
        
        return JSONResponse(
            status_code=403,
            content={
                "error": "Security validation failed",
                "message": "Session verification failed",
                "request_id": request_id
            }
        )
    
    # Standard error handling
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
    import os
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Run with uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug and settings.dev_auto_reload,
        log_level="debug" if settings.debug else "info",
        access_log=True,
        use_colors=True,
        workers=1 if settings.debug else settings.api_workers
    )
