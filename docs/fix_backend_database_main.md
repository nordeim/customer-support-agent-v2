# Complete Drop-in Replacement Files

## database.py - Complete Replacement

```python
"""
Database configuration and session management.
"""
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import logging
import os

from .config import settings

logger = logging.getLogger(__name__)

# Configure engine based on database type
if "sqlite" in settings.database_url:
    # SQLite specific configuration
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=settings.database_echo
    )
    logger.info("Using SQLite database")
else:
    # PostgreSQL or other databases
    engine = create_engine(
        settings.database_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=settings.database_echo
    )
    logger.info(f"Using database: {settings.database_url.split('@')[1] if '@' in settings.database_url else 'configured'}")

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db() -> Session:
    """
    Get database session.
    Dependency for FastAPI routes.
    
    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables."""
    try:
        logger.info("Initializing database...")
        
        # Import all models to ensure they're registered with Base
        from .models import memory, session, message  # Import all models
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables creation command executed")
        
        # Verify tables were created
        inspector = engine.dialect.get_inspector(engine)
        table_names = inspector.get_table_names()
        
        required_tables = ['sessions', 'messages', 'memories']
        missing_tables = [table for table in required_tables if table not in table_names]
        
        if missing_tables:
            logger.error(f"Missing tables after creation: {missing_tables}")
            logger.info(f"Available tables: {table_names}")
            raise Exception(f"Failed to create required tables: {missing_tables}")
        else:
            logger.info(f"Database tables created successfully: {table_names}")
            
        # Debug database state in development
        if settings.environment == "development":
            debug_database()
            
    except Exception as e:
        logger.error(f"Database initialization error: {e}", exc_info=True)
        raise


def cleanup_db() -> None:
    """Cleanup database connections."""
    logger.info("Cleaning up database connections...")
    engine.dispose()
    logger.info("Database cleanup complete")


def check_db_connection() -> bool:
    """Check database connection and basic functionality."""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            return result.fetchone()[0] == 1
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


def check_tables_exist() -> bool:
    """Check if required tables exist in the database."""
    try:
        inspector = engine.dialect.get_inspector(engine)
        table_names = inspector.get_table_names()
        
        required_tables = ['sessions', 'messages', 'memories']
        missing_tables = [table for table in required_tables if table not in table_names]
        
        if missing_tables:
            logger.error(f"Missing tables: {missing_tables}")
            return False
        
        logger.info(f"All required tables exist: {table_names}")
        return True
    except Exception as e:
        logger.error(f"Error checking tables: {e}")
        return False


def debug_database() -> None:
    """Debug function to check database state (development only)."""
    try:
        # Check database file
        if "sqlite" in settings.database_url:
            db_path = settings.database_url.replace("sqlite:///", "")
            if os.path.exists(db_path):
                logger.info(f"Database file exists: {db_path}")
                logger.info(f"Database file size: {os.path.getsize(db_path)} bytes")
            else:
                logger.warning(f"Database file does not exist: {db_path}")
                return
        
        # Check tables
        inspector = engine.dialect.get_inspector(engine)
        table_names = inspector.get_table_names()
        logger.info(f"Tables in database: {table_names}")
        
        # Check table schemas
        for table_name in table_names:
            columns = inspector.get_columns(table_name)
            logger.info(f"Table '{table_name}' columns: {[col['name'] for col in columns]}")
            
    except Exception as e:
        logger.error(f"Database debug error: {e}")
```

## main.py - Complete Replacement

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

## Key Improvements Implemented

### database.py Enhancements:

1. **Enhanced Table Verification**: Added explicit table existence checking after creation
2. **Robust Error Handling**: Comprehensive error handling with detailed logging
3. **Connection Validation**: Added `check_db_connection()` function for connection testing
4. **Table Existence Check**: Added `check_tables_exist()` function for startup validation
5. **Debug Functionality**: Added `debug_database()` function for development troubleshooting
6. **Environment-Specific Debug**: Debug functions only activate in development environment

### main.py Enhancements:

1. **Enhanced Startup Checks**: Integrated database and table verification in startup sequence
2. **Critical Component Validation**: Application fails startup if database tables are missing
3. **Specialized Error Handling**: Added specific handling for "no such table" errors
4. **Improved Error Recovery**: Enhanced error handling with database state checking
5. **Better Logging**: Added more detailed logging for troubleshooting
6. **Graceful Degradation**: Application handles missing components gracefully

## Validation Steps

1. **Database Initialization**: Tables are now verified to exist after creation
2. **Startup Validation**: Critical components are checked before application starts
3. **Error Handling**: Database errors are caught and handled appropriately
4. **Debug Information**: Development environment provides detailed debugging information
5. **Production Safety**: Debug features are disabled in production

These files are complete drop-in replacements that maintain all existing functionality while adding robust database initialization verification and enhanced error handling to resolve the "no such table: sessions" issue.

---


