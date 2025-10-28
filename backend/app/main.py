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
        logger.info(f"Starting {settings.app_name} v{settings.version}")
        logger.info(f"Environment: {settings.environment}")
        logger.info("=" * 60)
        
        # Initialize database
        logger.info("Initializing database...")
        init_db()
        
        # Initialize cache service
        logger.info("Initializing cache service...")
        cache_service = CacheService()
        app.state.cache = cache_service
        
        # Test cache connection
        if await cache_service.ping():
            logger.info("✓ Cache service connected")
        else:
            logger.warning("✗ Cache service unavailable - running without cache")
        
        # Initialize telemetry
        if settings.enable_telemetry:
            setup_telemetry(app)
            logger.info("✓ Telemetry initialized")
        
        # Initialize the AI agent
        logger.info("Initializing AI agent...")
        agent = CustomerSupportAgent()
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
    
    # Check database
    try:
        from .database import get_db
        db = next(get_db())
        db.execute("SELECT 1")
        db.close()
        checks.append("Database: ✓")
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        checks.append("Database: ✗")
    
    # Check Redis
    if app.state.cache.enabled:
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
    
    logger.info(f"Startup checks: {', '.join(checks)}")


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
    
    logger.error(
        f"Unhandled exception in request {request_id}: {exc}",
        exc_info=True,
        extra={
            "path": request.url.path,
            "method": request.method,
            "client": request.client.host if request.client else "unknown"
        }
    )
    
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
