"""
Database configuration and session management with production optimizations.
Supports both SQLite and PostgreSQL with async support and enhanced error handling.

Version: 2.0.0 (Async-ready with comprehensive health monitoring)
"""
from sqlalchemy import create_engine, text, inspect, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import StaticPool, QueuePool, NullPool
from sqlalchemy.exc import (
    SQLAlchemyError,
    DisconnectionError,
    OperationalError,
    DatabaseError
)
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)
import logging
import os
import time
import asyncio
import threading
from contextlib import contextmanager, asynccontextmanager
from typing import Generator, Optional, Dict, Any, AsyncGenerator
from pathlib import Path

from .config import settings

logger = logging.getLogger(__name__)

# Create declarative base
Base = declarative_base()

# Global engines and session factories
_engine = None
_async_engine = None
_SessionLocal = None
_AsyncSessionLocal = None
_init_lock = threading.Lock()
_async_init_lock = asyncio.Lock()
_initialized = False
_async_initialized = False


def _enable_sqlite_wal_mode(dbapi_connection, connection_record):
    """Enable WAL mode for SQLite for better concurrency."""
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
        cursor.close()
        logger.debug("SQLite WAL mode and optimizations enabled")
    except Exception as e:
        logger.warning(f"Failed to enable SQLite optimizations: {e}")


def _ping_connection(dbapi_connection, connection_record, connection_proxy):
    """Ping connection on checkout to ensure it's alive."""
    try:
        dbapi_connection.cursor().execute("SELECT 1")
    except Exception as e:
        logger.warning(f"Connection ping failed, invalidating connection: {e}")
        raise DisconnectionError("Connection lost")


def _log_connection_checkout(dbapi_connection, connection_record, connection_proxy):
    """Log connection checkout for monitoring."""
    logger.debug(f"Connection checked out from pool")


def _log_connection_checkin(dbapi_connection, connection_record):
    """Log connection checkin for monitoring."""
    logger.debug(f"Connection returned to pool")


def create_database_engine() -> None:
    """
    Create synchronous database engine based on configuration.
    Thread-safe with initialization lock.
    """
    global _engine, _SessionLocal, _initialized
    
    # Double-checked locking pattern
    if _initialized and _engine is not None:
        return
    
    with _init_lock:
        # Check again inside lock
        if _initialized and _engine is not None:
            return
        
        try:
            logger.info("Creating database engine...")
            
            if settings.database_is_sqlite:
                # SQLite configuration with WAL mode
                db_path = settings.database_url.replace('sqlite:///', '')
                
                # Ensure directory exists
                if not os.path.isabs(db_path):
                    db_dir = os.path.dirname(db_path)
                    if db_dir:
                        Path(db_dir).mkdir(parents=True, exist_ok=True)
                
                _engine = create_engine(
                    settings.database_url,
                    connect_args={
                        "check_same_thread": False,
                        "timeout": 20,
                        "isolation_level": None
                    },
                    poolclass=StaticPool,
                    echo=settings.database_echo,
                    pool_pre_ping=True,
                    # Retry on disconnect
                    pool_recycle=3600
                )
                
                # Enable WAL mode
                event.listen(_engine, "connect", _enable_sqlite_wal_mode)
                
                logger.info(f"SQLite database engine created: {db_path}")
                
            elif settings.database_is_postgresql:
                # PostgreSQL configuration with connection pooling
                _engine = create_engine(
                    settings.database_url,
                    poolclass=QueuePool,
                    pool_size=settings.database_pool_size,
                    max_overflow=settings.database_pool_overflow,
                    pool_timeout=settings.database_pool_timeout,
                    pool_recycle=settings.database_pool_recycle,
                    pool_pre_ping=True,
                    echo=settings.database_echo,
                    connect_args={
                        "application_name": settings.app_name,
                        "connect_timeout": 10,
                        "options": "-c timezone=UTC"
                    },
                    # Connection pool optimization
                    pool_use_lifo=True,  # LIFO for better connection reuse
                )
                
                # PostgreSQL-specific event listeners
                @event.listens_for(_engine, "connect")
                def set_postgresql_search_path(dbapi_connection, connection_record):
                    try:
                        with dbapi_connection.cursor() as cursor:
                            cursor.execute("SET search_path TO public")
                    except Exception as e:
                        logger.warning(f"Failed to set search_path: {e}")
                
                logger.info(
                    f"PostgreSQL database engine created "
                    f"(pool_size={settings.database_pool_size}, "
                    f"max_overflow={settings.database_pool_overflow})"
                )
                
            else:
                # Generic database configuration
                _engine = create_engine(
                    settings.database_url,
                    pool_size=settings.database_pool_size,
                    max_overflow=settings.database_pool_overflow,
                    pool_timeout=settings.database_pool_timeout,
                    pool_pre_ping=True,
                    echo=settings.database_echo
                )
                logger.info("Generic database engine created")
            
            # Add connection monitoring
            event.listen(_engine, "checkout", _ping_connection)
            
            if settings.debug:
                event.listen(_engine, "checkout", _log_connection_checkout)
                event.listen(_engine, "checkin", _log_connection_checkin)
            
            # Create session factory
            _SessionLocal = scoped_session(
                sessionmaker(
                    autocommit=False,
                    autoflush=False,
                    bind=_engine,
                    expire_on_commit=False
                )
            )
            
            _initialized = True
            logger.info("✓ Database engine created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}", exc_info=True)
            _initialized = False
            raise


async def create_async_database_engine() -> None:
    """
    Create asynchronous database engine for async operations.
    Async-safe with initialization lock.
    """
    global _async_engine, _AsyncSessionLocal, _async_initialized
    
    # Double-checked locking pattern (async)
    if _async_initialized and _async_engine is not None:
        return
    
    async with _async_init_lock:
        # Check again inside lock
        if _async_initialized and _async_engine is not None:
            return
        
        try:
            logger.info("Creating async database engine...")
            
            if settings.database_is_sqlite:
                # SQLite async with aiosqlite
                async_url = settings.database_url.replace('sqlite:///', 'sqlite+aiosqlite:///')
                
                db_path = settings.database_url.replace('sqlite:///', '')
                if not os.path.isabs(db_path):
                    db_dir = os.path.dirname(db_path)
                    if db_dir:
                        Path(db_dir).mkdir(parents=True, exist_ok=True)
                
                _async_engine = create_async_engine(
                    async_url,
                    connect_args={
                        "check_same_thread": False,
                        "timeout": 20
                    },
                    poolclass=NullPool,  # aiosqlite doesn't support connection pooling well
                    echo=settings.database_echo
                )
                
                logger.info(f"Async SQLite database engine created: {db_path}")
                
            elif settings.database_is_postgresql:
                # PostgreSQL async with asyncpg
                async_url = settings.database_url.replace('postgresql://', 'postgresql+asyncpg://')
                
                _async_engine = create_async_engine(
                    async_url,
                    pool_size=settings.database_pool_size,
                    max_overflow=settings.database_pool_overflow,
                    pool_timeout=settings.database_pool_timeout,
                    pool_recycle=settings.database_pool_recycle,
                    pool_pre_ping=True,
                    echo=settings.database_echo,
                    connect_args={
                        "server_settings": {
                            "application_name": settings.app_name,
                            "timezone": "UTC"
                        },
                        "timeout": 10
                    }
                )
                
                logger.info(
                    f"Async PostgreSQL database engine created "
                    f"(pool_size={settings.database_pool_size})"
                )
                
            else:
                logger.warning("Async engine only supports SQLite and PostgreSQL")
                _async_initialized = False
                return
            
            # Create async session factory
            _AsyncSessionLocal = async_sessionmaker(
                _async_engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
            
            _async_initialized = True
            logger.info("✓ Async database engine created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create async database engine: {e}", exc_info=True)
            _async_initialized = False
            raise


def get_engine():
    """Get the database engine, creating it if necessary."""
    if _engine is None:
        create_database_engine()
    return _engine


async def get_async_engine():
    """Get the async database engine, creating it if necessary."""
    if _async_engine is None:
        await create_async_database_engine()
    return _async_engine


def get_db() -> Generator[Session, None, None]:
    """
    Get database session with proper error handling (synchronous).
    
    Yields:
        Database session
    """
    if _SessionLocal is None:
        create_database_engine()
    
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    db = _SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    except Exception as e:
        logger.error(f"Unexpected database error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get async database session with proper error handling.
    
    Yields:
        Async database session
    """
    if _AsyncSessionLocal is None:
        await create_async_database_engine()
    
    if _AsyncSessionLocal is None:
        raise RuntimeError("Async database not initialized")
    
    async with _AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except SQLAlchemyError as e:
            logger.error(f"Async database session error: {e}")
            await session.rollback()
            raise
        except Exception as e:
            logger.error(f"Unexpected async database error: {e}")
            await session.rollback()
            raise


@contextmanager
def get_db_context():
    """
    Context manager for database sessions (synchronous).
    
    Returns:
        Database session
    """
    if _SessionLocal is None:
        create_database_engine()
    
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    db = _SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database context error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@asynccontextmanager
async def get_async_db_context():
    """
    Async context manager for database sessions.
    
    Returns:
        Async database session
    """
    if _AsyncSessionLocal is None:
        await create_async_database_engine()
    
    if _AsyncSessionLocal is None:
        raise RuntimeError("Async database not initialized")
    
    async with _AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            logger.error(f"Async database context error: {e}")
            await session.rollback()
            raise


def init_db() -> None:
    """
    Initialize database tables with enhanced error handling.
    Thread-safe with proper locking.
    """
    global _initialized
    
    with _init_lock:
        try:
            logger.info("Initializing database...")
            
            # Create engine if not exists
            if _engine is None:
                create_database_engine()
            
            if _engine is None:
                raise RuntimeError("Failed to create database engine")
            
            # Import all models to register with Base
            try:
                from .models import memory, session, message
                logger.debug("Models imported successfully")
            except ImportError as e:
                logger.error(f"Failed to import models: {e}")
                raise
            
            # Database-specific initialization
            if settings.database_is_postgresql:
                logger.info("Performing PostgreSQL-specific initialization...")
                
                try:
                    with _engine.connect() as conn:
                        # Create extensions
                        try:
                            conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
                            conn.execute(text('CREATE EXTENSION IF NOT EXISTS "pg_trgm"'))
                            logger.info("✓ PostgreSQL extensions created")
                        except OperationalError as e:
                            logger.warning(f"Failed to create PostgreSQL extensions (may already exist): {e}")
                        
                        # Set timezone
                        conn.execute(text("SET timezone = 'UTC'"))
                        conn.commit()
                        logger.info("✓ PostgreSQL timezone set to UTC")
                        
                except Exception as e:
                    logger.warning(f"PostgreSQL initialization warning: {e}")
            
            # Create all tables
            logger.info("Creating database tables...")
            start_time = time.time()
            
            Base.metadata.create_all(bind=_engine, checkfirst=True)
            
            creation_time = time.time() - start_time
            logger.info(f"✓ Database tables created in {creation_time:.2f}s")
            
            # Verify tables
            inspector = inspect(_engine)
            table_names = inspector.get_table_names()
            
            required_tables = ['sessions', 'messages', 'memories']
            missing_tables = [table for table in required_tables if table not in table_names]
            
            if missing_tables:
                logger.error(f"Missing tables after creation: {missing_tables}")
                logger.info(f"Available tables: {table_names}")
                
                # Attempt individual table creation
                for table_name in missing_tables:
                    try:
                        logger.info(f"Attempting to create table '{table_name}' individually...")
                        
                        if table_name == 'sessions':
                            from .models.session import Session as SessionModel
                            SessionModel.__table__.create(bind=_engine, checkfirst=True)
                        elif table_name == 'messages':
                            from .models.message import Message as MessageModel
                            MessageModel.__table__.create(bind=_engine, checkfirst=True)
                        elif table_name == 'memories':
                            from .models.memory import Memory as MemoryModel
                            MemoryModel.__table__.create(bind=_engine, checkfirst=True)
                        
                        logger.info(f"✓ Table '{table_name}' created")
                        
                    except Exception as e:
                        logger.error(f"Failed to create table '{table_name}': {e}")
                
                # Final verification
                inspector = inspect(_engine)
                table_names = inspector.get_table_names()
                missing_tables = [table for table in required_tables if table not in table_names]
                
                if missing_tables:
                    raise RuntimeError(f"Failed to create required tables: {missing_tables}")
            
            logger.info(f"Database tables: {table_names}")
            
            # Debug in development
            if settings.is_development and settings.debug:
                debug_database()
            
            _initialized = True
            logger.info("✓ Database initialization complete")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}", exc_info=True)
            _initialized = False
            raise


async def init_async_db() -> None:
    """
    Initialize async database.
    Async-safe with proper locking.
    """
    global _async_initialized
    
    async with _async_init_lock:
        try:
            logger.info("Initializing async database...")
            
            # Create async engine if not exists
            if _async_engine is None:
                await create_async_database_engine()
            
            if _async_engine is None:
                logger.warning("Async database engine not available")
                return
            
            # Import models
            try:
                from .models import memory, session, message
                logger.debug("Models imported for async database")
            except ImportError as e:
                logger.error(f"Failed to import models: {e}")
                raise
            
            # Create tables
            logger.info("Creating async database tables...")
            start_time = time.time()
            
            async with _async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all, checkfirst=True)
            
            creation_time = time.time() - start_time
            logger.info(f"✓ Async database tables created in {creation_time:.2f}s")
            
            _async_initialized = True
            logger.info("✓ Async database initialization complete")
            
        except Exception as e:
            logger.error(f"Async database initialization failed: {e}", exc_info=True)
            _async_initialized = False
            raise


def cleanup_db() -> None:
    """
    Cleanup database connections with proper resource management.
    Thread-safe with graceful shutdown.
    """
    global _engine, _SessionLocal, _initialized
    
    with _init_lock:
        try:
            logger.info("Cleaning up database connections...")
            
            if _SessionLocal:
                try:
                    _SessionLocal.remove()
                    logger.info("✓ Database sessions cleaned up")
                except Exception as e:
                    logger.error(f"Error cleaning up sessions: {e}")
                finally:
                    _SessionLocal = None
            
            if _engine:
                try:
                    # Wait for connections to be returned (graceful shutdown)
                    pool = _engine.pool
                    if hasattr(pool, 'checkedout'):
                        checked_out = pool.checkedout()
                        if checked_out > 0:
                            logger.info(f"Waiting for {checked_out} connections to be returned...")
                            time.sleep(1)
                    
                    _engine.dispose()
                    logger.info("✓ Database engine disposed")
                except Exception as e:
                    logger.error(f"Error disposing engine: {e}")
                finally:
                    _engine = None
            
            _initialized = False
            logger.info("✓ Database cleanup complete")
            
        except Exception as e:
            logger.error(f"Database cleanup error: {e}")


async def cleanup_async_db() -> None:
    """
    Cleanup async database connections.
    Async-safe with graceful shutdown.
    """
    global _async_engine, _AsyncSessionLocal, _async_initialized
    
    async with _async_init_lock:
        try:
            logger.info("Cleaning up async database connections...")
            
            if _AsyncSessionLocal:
                _AsyncSessionLocal = None
                logger.info("✓ Async database sessions cleaned up")
            
            if _async_engine:
                try:
                    await _async_engine.dispose()
                    logger.info("✓ Async database engine disposed")
                except Exception as e:
                    logger.error(f"Error disposing async engine: {e}")
                finally:
                    _async_engine = None
            
            _async_initialized = False
            logger.info("✓ Async database cleanup complete")
            
        except Exception as e:
            logger.error(f"Async database cleanup error: {e}")


def check_db_connection(max_retries: int = 3, retry_delay: float = 1.0) -> bool:
    """
    Check database connection with retry logic.
    
    Args:
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        True if connection is healthy
    """
    if _engine is None:
        logger.error("Database engine not initialized")
        return False
    
    for attempt in range(max_retries):
        try:
            with _engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                if result.fetchone()[0] == 1:
                    logger.debug("Database connection check passed")
                    return True
                
        except (DisconnectionError, OperationalError) as e:
            logger.warning(
                f"Database connection check failed "
                f"(attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        except Exception as e:
            logger.error(f"Unexpected database connection error: {e}")
            break
    
    logger.error("Database connection check failed after all retries")
    return False


async def check_async_db_connection(max_retries: int = 3, retry_delay: float = 1.0) -> bool:
    """
    Check async database connection with retry logic.
    
    Args:
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        True if connection is healthy
    """
    if _async_engine is None:
        logger.error("Async database engine not initialized")
        return False
    
    for attempt in range(max_retries):
        try:
            async with _async_engine.connect() as connection:
                result = await connection.execute(text("SELECT 1"))
                row = result.fetchone()
                if row and row[0] == 1:
                    logger.debug("Async database connection check passed")
                    return True
                
        except (DisconnectionError, OperationalError) as e:
            logger.warning(
                f"Async database connection check failed "
                f"(attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
        except Exception as e:
            logger.error(f"Unexpected async database connection error: {e}")
            break
    
    logger.error("Async database connection check failed after all retries")
    return False


def check_tables_exist() -> bool:
    """
    Check if required tables exist.
    
    Returns:
        True if all required tables exist
    """
    if _engine is None:
        logger.error("Database engine not initialized")
        return False
    
    try:
        inspector = inspect(_engine)
        table_names = inspector.get_table_names()
        
        required_tables = ['sessions', 'messages', 'memories']
        missing_tables = [table for table in required_tables if table not in table_names]
        
        if missing_tables:
            logger.error(f"Missing tables: {missing_tables}")
            return False
        
        logger.debug(f"All required tables exist: {required_tables}")
        return True
        
    except Exception as e:
        logger.error(f"Error checking tables: {e}")
        return False


def get_database_info() -> Dict[str, Any]:
    """
    Get database information for monitoring.
    
    Returns:
        Dictionary with database information
    """
    if _engine is None:
        return {"status": "not_initialized"}
    
    try:
        info = {
            "status": "connected",
            "url": settings.database_url.split('@')[-1] if '@' in settings.database_url else "sqlite",
            "type": "postgresql" if settings.database_is_postgresql else "sqlite",
            "initialized": _initialized,
            "async_initialized": _async_initialized
        }
        
        # Pool information
        pool = _engine.pool
        if hasattr(pool, 'size'):
            info.update({
                "pool_size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow()
            })
        
        # PostgreSQL-specific info
        if settings.database_is_postgresql:
            try:
                with _engine.connect() as conn:
                    result = conn.execute(text("SELECT version()"))
                    info["version"] = result.fetchone()[0]
                    
                    result = conn.execute(
                        text("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'")
                    )
                    info["table_count"] = result.fetchone()[0]
            except Exception as e:
                logger.warning(f"Failed to get PostgreSQL info: {e}")
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        return {"status": "error", "error": str(e)}


def debug_database() -> None:
    """Debug function to check database state (development only)."""
    if not settings.is_development:
        return
    
    try:
        logger.info("=== Database Debug Information ===")
        
        # Check SQLite file
        if settings.database_is_sqlite:
            db_path = settings.database_url.replace("sqlite:///", "")
            if os.path.exists(db_path):
                file_size = os.path.getsize(db_path)
                logger.info(f"SQLite file: {db_path} ({file_size:,} bytes)")
            else:
                logger.warning(f"SQLite file does not exist: {db_path}")
        
        # Check tables
        if _engine:
            inspector = inspect(_engine)
            table_names = inspector.get_table_names()
            logger.info(f"Tables: {table_names}")
            
            for table_name in table_names:
                columns = inspector.get_columns(table_name)
                logger.info(f"  {table_name}: {[col['name'] for col in columns]}")
                
                try:
                    with _engine.connect() as conn:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                        count = result.fetchone()[0]
                        logger.info(f"  {table_name} rows: {count}")
                except Exception as e:
                    logger.warning(f"  Could not count {table_name}: {e}")
        
        logger.info("=== End Database Debug ===")
        
    except Exception as e:
        logger.error(f"Database debug error: {e}")


def get_database_health() -> Dict[str, Any]:
    """
    Get comprehensive database health status.
    
    Returns:
        Dictionary with health information
    """
    health = {
        "status": "unhealthy",
        "connection": False,
        "tables": False,
        "timestamp": time.time()
    }
    
    try:
        # Check connection
        if check_db_connection():
            health["connection"] = True
            
            # Check tables
            if check_tables_exist():
                health["tables"] = True
                health["status"] = "healthy"
        
        # Add info
        health["info"] = get_database_info()
        
    except Exception as e:
        health["error"] = str(e)
        logger.error(f"Database health check error: {e}")
    
    return health


# Auto-initialize on module import (non-testing environments)
if not os.environ.get("TESTING"):
    try:
        create_database_engine()
    except Exception as e:
        logger.warning(f"Failed to auto-initialize database on import: {e}")
