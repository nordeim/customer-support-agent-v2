"""
Database configuration and session management with production optimizations.
Supports both SQLite and PostgreSQL with enhanced error handling and monitoring.
"""
from sqlalchemy import create_engine, text, inspect, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import StaticPool, QueuePool
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
import logging
import os
import time
from contextlib import contextmanager
from typing import Generator, Optional

from .config import settings

logger = logging.getLogger(__name__)

# Create declarative base
Base = declarative_base()

# Global engine and session factory
engine = None
SessionLocal = None


def create_database_engine() -> None:
    """
    Create database engine based on configuration.
    Supports both SQLite and PostgreSQL with production optimizations.
    """
    global engine, SessionLocal
    
    try:
        if "sqlite" in settings.database_url:
            # SQLite specific configuration
            engine = create_engine(
                settings.database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=settings.database_echo,
                # SQLite-specific optimizations
                pool_pre_ping=True,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 20,
                    "isolation_level": None
                }
            )
            logger.info("Using SQLite database")
            
        elif "postgresql" in settings.database_url:
            # PostgreSQL specific configuration with connection pooling
            engine = create_engine(
                settings.database_url,
                # Connection pool settings
                poolclass=QueuePool,
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_pool_overflow,
                pool_timeout=settings.database_pool_timeout,
                pool_recycle=settings.database_pool_recycle,
                pool_pre_ping=True,
                # PostgreSQL-specific settings
                echo=settings.database_echo,
                connect_args={
                    "application_name": settings.app_name,
                    "connect_timeout": 10,
                    "command_timeout": 30,
                    # Optimize for production
                    "options": "-c timezone=UTC"
                }
            )
            logger.info(f"Using PostgreSQL database with pool size: {settings.database_pool_size}")
            
            # Add PostgreSQL-specific event listeners
            @event.listens_for(engine, "connect")
            def set_postgresql_search_path(dbapi_connection, connection_record):
                with dbapi_connection.cursor() as cursor:
                    cursor.execute("SET search_path TO public")
            
            @event.listens_for(engine, "checkout")
            def receive_checkout(dbapi_connection, connection_record, connection_proxy):
                # Log connection checkout for monitoring
                logger.debug("Database connection checked out from pool")
                
        else:
            # Other databases (MySQL, etc.)
            engine = create_engine(
                settings.database_url,
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_pool_overflow,
                pool_timeout=settings.database_pool_timeout,
                pool_pre_ping=True,
                echo=settings.database_echo
            )
            logger.info(f"Using database: {settings.database_url.split('@')[1] if '@' in settings.database_url else 'configured'}")
        
        # Create session factory
        SessionLocal = scoped_session(
            sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=engine,
                # Production optimizations
                expire_on_commit=False
            )
        )
        
        logger.info("Database engine created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}", exc_info=True)
        raise


def get_db() -> Generator[Session, None, None]:
    """
    Get database session with proper error handling.
    
    Yields:
        Database session
    """
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    db = SessionLocal()
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


@contextmanager
def get_db_context():
    """
    Context manager for database sessions.
    
    Returns:
        Database session
    """
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database context error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize database tables with enhanced error handling and monitoring.
    """
    global engine, SessionLocal
    
    try:
        logger.info("Initializing database...")
        
        # Create engine if not exists
        if engine is None:
            create_database_engine()
        
        # Import all models to ensure they're registered with Base
        from .models import memory, session, message
        
        # Database-specific initialization
        if settings.database_is_postgresql:
            # PostgreSQL-specific setup
            logger.info("Performing PostgreSQL-specific initialization...")
            
            with engine.connect() as conn:
                # Create extensions
                try:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\""))
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"pg_trgm\""))
                    logger.info("PostgreSQL extensions created")
                except Exception as e:
                    logger.warning(f"Failed to create PostgreSQL extensions: {e}")
                
                # Set timezone
                conn.execute(text("SET timezone = 'UTC'"))
                conn.commit()
        
        # Create all tables
        logger.info("Creating database tables...")
        start_time = time.time()
        
        Base.metadata.create_all(bind=engine)
        
        creation_time = time.time() - start_time
        logger.info(f"Database tables created successfully in {creation_time:.2f} seconds")
        
        # Verify tables were created
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        required_tables = ['sessions', 'messages', 'memories']
        missing_tables = [table for table in required_tables if table not in table_names]
        
        if missing_tables:
            logger.error(f"Missing tables after creation: {missing_tables}")
            logger.info(f"Available tables: {table_names}")
            
            # Try to create missing tables individually
            for table_name in missing_tables:
                try:
                    logger.info(f"Attempting to create table {table_name} individually...")
                    
                    if table_name == 'sessions':
                        from .models.session import Session as SessionModel
                        SessionModel.__table__.create(bind=engine, checkfirst=True)
                    elif table_name == 'messages':
                        from .models.message import Message as MessageModel
                        MessageModel.__table__.create(bind=engine, checkfirst=True)
                    elif table_name == 'memories':
                        from .models.memory import Memory as MemoryModel
                        MemoryModel.__table__.create(bind=engine, checkfirst=True)
                    
                    logger.info(f"Table {table_name} created successfully")
                    
                except Exception as e:
                    logger.error(f"Failed to create table {table_name}: {e}")
            
            # Final verification
            inspector = inspect(engine)
            table_names = inspector.get_table_names()
            missing_tables = [table for table in required_tables if table not in table_names]
            
            if missing_tables:
                raise Exception(f"Failed to create required tables: {missing_tables}")
        
        # Log table information
        logger.info(f"Database tables: {table_names}")
        
        # Debug database state in development
        if settings.environment == "development":
            debug_database()
        
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Database initialization error: {e}", exc_info=True)
        raise


def cleanup_db() -> None:
    """
    Cleanup database connections with proper resource management.
    """
    global engine, SessionLocal
    
    try:
        logger.info("Cleaning up database connections...")
        
        if SessionLocal:
            SessionLocal.remove()
            SessionLocal = None
            logger.info("Database sessions cleaned up")
        
        if engine:
            engine.dispose()
            engine = None
            logger.info("Database engine disposed")
        
        logger.info("Database cleanup complete")
        
    except Exception as e:
        logger.error(f"Error during database cleanup: {e}")


def check_db_connection() -> bool:
    """
    Check database connection and basic functionality with retry logic.
    
    Returns:
        True if connection is healthy, False otherwise
    """
    if engine is None:
        logger.error("Database engine not initialized")
        return False
    
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            with engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                if result.fetchone()[0] == 1:
                    logger.debug("Database connection check passed")
                    return True
                
        except (SQLAlchemyError, DisconnectionError) as e:
            logger.warning(f"Database connection check failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
        except Exception as e:
            logger.error(f"Unexpected database connection error: {e}")
            break
    
    logger.error("Database connection check failed after all retries")
    return False


def check_tables_exist() -> bool:
    """
    Check if required tables exist in the database.
    
    Returns:
        True if all required tables exist, False otherwise
    """
    if engine is None:
        logger.error("Database engine not initialized")
        return False
    
    try:
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        required_tables = ['sessions', 'messages', 'memories']
        missing_tables = [table for table in required_tables if table not in table_names]
        
        if missing_tables:
            logger.error(f"Missing tables: {missing_tables}")
            return False
        
        logger.debug(f"All required tables exist: {table_names}")
        return True
        
    except Exception as e:
        logger.error(f"Error checking tables: {e}")
        return False


def get_database_info() -> dict:
    """
    Get database information for monitoring.
    
    Returns:
        Dictionary with database information
    """
    if engine is None:
        return {"status": "not_initialized"}
    
    try:
        info = {
            "status": "connected",
            "url": settings.database_url.split('@')[-1] if '@' in settings.database_url else "sqlite",
            "pool_size": getattr(engine.pool, 'size', 'N/A'),
            "checked_in": getattr(engine.pool, 'checkedin', 'N/A'),
            "checked_out": getattr(engine.pool, 'checkedout', 'N/A')
        }
        
        if settings.database_is_postgresql:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                info["version"] = result.fetchone()[0]
                
                result = conn.execute(text("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'"))
                info["table_count"] = result.fetchone()[0]
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        return {"status": "error", "error": str(e)}


def debug_database() -> None:
    """
    Debug function to check database state (development only).
    """
    if not settings.environment == "development":
        return
    
    try:
        logger.info("=== Database Debug Information ===")
        
        # Check database file for SQLite
        if settings.database_is_sqlite:
            db_path = settings.database_url.replace("sqlite:///", "")
            if os.path.exists(db_path):
                file_size = os.path.getsize(db_path)
                logger.info(f"SQLite database file: {db_path}")
                logger.info(f"Database file size: {file_size:,} bytes")
            else:
                logger.warning(f"SQLite database file does not exist: {db_path}")
                return
        
        # Check tables
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        logger.info(f"Tables in database: {table_names}")
        
        # Check table schemas
        for table_name in table_names:
            columns = inspector.get_columns(table_name)
            logger.info(f"Table '{table_name}' columns: {[col['name'] for col in columns]}")
            
            # Check row counts
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    count = result.fetchone()[0]
                    logger.info(f"Table '{table_name}' rows: {count}")
            except Exception as e:
                logger.warning(f"Could not get row count for table '{table_name}': {e}")
        
        logger.info("=== End Database Debug Information ===")
        
    except Exception as e:
        logger.error(f"Database debug error: {e}")


def run_database_migrations() -> None:
    """
    Run database migrations if needed.
    This is a placeholder for future migration implementation.
    """
    logger.info("Checking for database migrations...")
    
    # TODO: Implement proper migration system (Alembic)
    # For now, just ensure tables exist
    if not check_tables_exist():
        logger.info("Running database migration...")
        init_db()
    else:
        logger.info("Database migrations up to date")


# Health check functions
def get_database_health() -> dict:
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
        
        # Add additional info
        health["info"] = get_database_info()
        
    except Exception as e:
        health["error"] = str(e)
        logger.error(f"Database health check error: {e}")
    
    return health


# Initialize database on module import if not in testing
if not os.environ.get("TESTING"):
    try:
        create_database_engine()
    except Exception as e:
        logger.warning(f"Failed to create database engine on import: {e}")
