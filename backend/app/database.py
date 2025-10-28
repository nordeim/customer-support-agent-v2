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
