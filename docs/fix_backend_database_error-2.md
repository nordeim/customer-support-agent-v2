# Analysis of Model Import Error

## Executive Summary

The application is now failing with a new error: `cannot import name 'SessionModel' from 'app.models.session'` and `cannot import name 'MessageModel' from 'app.models.message'`. This indicates that the model classes have different names than what we're trying to import. The debug output shows that the models are being imported successfully, but they're not being registered with the Base declarative class, and we're using incorrect class names for individual table creation.

## Detailed Analysis

### Error Breakdown

```
2025-10-28 18:39:44,305 - app.database - ERROR - Failed to create table sessions: cannot import name 'SessionModel' from 'app.models.session' (/Home1/project/customer-support-agent-v2/backend/app/models/session.py)
2025-10-28 18:39:44,305 - app.database - ERROR - Failed to create table messages: cannot import name 'MessageModel' from 'app.models.message' (/Home1/project/customer-support-agent-v2/backend/app/models/message.py)
```

The key observations from the debug output are:

1. **No Registered Models**: The `Registered models:` and `Tables to be created:` sections are empty, indicating that no models are being registered with the Base declarative class.
2. **Successful Imports**: The models are being imported successfully (`Memory model imported successfully`, etc.).
3. **Incorrect Class Names**: We're trying to import `SessionModel` and `MessageModel`, but these might not be the correct class names.

### Root Cause Analysis

The most likely causes for this issue are:

1. **Incorrect Class Names**: The model classes might have different names than `SessionModel` and `MessageModel`.
2. **Model Registration Issues**: The models might not be properly inheriting from the Base declarative class.
3. **Import Structure Issues**: The models might be structured differently than expected.

Since we don't have access to the model files, we need to investigate further. However, we can modify our approach to:

1. Use a more dynamic approach to find the correct class names
2. Add more debugging to understand the model structure
3. Use a more robust method to create tables individually

## Implementation Plan

1. Add a function to dynamically discover model classes in the imported modules
2. Modify the individual table creation to use the discovered class names
3. Add more detailed debugging to understand the model structure
4. Implement a fallback approach if individual table creation fails

## Implementation

### Updated database.py with Dynamic Model Discovery

```python
"""
Database configuration and session management.
"""
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import logging
import os
import inspect as py_inspect

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


def debug_registered_models():
    """Debug function to list all registered models."""
    try:
        logger.info("Registered models:")
        for class_name in Base.registry._class_registry.keys():
            if not class_name.startswith('_'):
                logger.info(f"  - {class_name}")
        
        logger.info("Tables to be created:")
        for table_name in Base.metadata.tables.keys():
            logger.info(f"  - {table_name}")
    except Exception as e:
        logger.error(f"Error debugging registered models: {e}")


def discover_model_classes(module):
    """Discover SQLAlchemy model classes in a module."""
    model_classes = []
    try:
        for name, obj in py_inspect.getmembers(module):
            if (py_inspect.isclass(obj) and 
                hasattr(obj, '__tablename__') and 
                hasattr(obj, '__table__')):
                model_classes.append((name, obj))
                logger.info(f"Found model class: {name} with table: {obj.__tablename__}")
    except Exception as e:
        logger.error(f"Error discovering model classes: {e}")
    return model_classes


def init_db() -> None:
    """Initialize database tables."""
    try:
        logger.info("Initializing database...")
        
        # Debug registered models before import
        debug_registered_models()
        
        # Import all models to ensure they're registered with Base
        memory_models = []
        session_models = []
        message_models = []
        
        try:
            logger.info("Importing memory model...")
            from .models import memory
            memory_models = discover_model_classes(memory)
            logger.info("Memory model imported successfully")
        except Exception as e:
            logger.error(f"Failed to import memory model: {e}")
        
        try:
            logger.info("Importing session model...")
            from .models import session
            session_models = discover_model_classes(session)
            logger.info("Session model imported successfully")
        except Exception as e:
            logger.error(f"Failed to import session model: {e}")
        
        try:
            logger.info("Importing message model...")
            from .models import message
            message_models = discover_model_classes(message)
            logger.info("Message model imported successfully")
        except Exception as e:
            logger.error(f"Failed to import message model: {e}")
        
        # Debug registered models after import
        debug_registered_models()
        
        # Create all tables
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables creation command executed")
        
        # Verify tables were created
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        required_tables = ['sessions', 'messages', 'memories']
        missing_tables = [table for table in required_tables if table not in table_names]
        
        if missing_tables:
            logger.error(f"Missing tables after creation: {missing_tables}")
            logger.info(f"Available tables: {table_names}")
            
            # Try to create missing tables individually using discovered model classes
            for table_name in missing_tables:
                try:
                    logger.info(f"Attempting to create table {table_name} individually...")
                    
                    if table_name == 'sessions' and session_models:
                        for class_name, model_class in session_models:
                            if model_class.__tablename__ == table_name:
                                model_class.__table__.create(bind=engine, checkfirst=True)
                                logger.info(f"Table {table_name} created using class {class_name}")
                                break
                    elif table_name == 'messages' and message_models:
                        for class_name, model_class in message_models:
                            if model_class.__tablename__ == table_name:
                                model_class.__table__.create(bind=engine, checkfirst=True)
                                logger.info(f"Table {table_name} created using class {class_name}")
                                break
                    elif table_name == 'memories' and memory_models:
                        for class_name, model_class in memory_models:
                            if model_class.__tablename__ == table_name:
                                model_class.__table__.create(bind=engine, checkfirst=True)
                                logger.info(f"Table {table_name} created using class {class_name}")
                                break
                    else:
                        logger.warning(f"No matching model class found for table {table_name}")
                        
                except Exception as e:
                    logger.error(f"Failed to create table {table_name}: {e}")
            
            # Check again after individual creation attempts
            inspector = inspect(engine)
            table_names = inspector.get_table_names()
            missing_tables = [table for table in required_tables if table not in table_names]
            
            if missing_tables:
                logger.error(f"Still missing tables after individual creation: {missing_tables}")
                
                # As a last resort, try to create tables using raw SQL
                logger.info("Attempting to create missing tables using raw SQL...")
                try:
                    if 'sessions' in missing_tables:
                        with engine.connect() as conn:
                            conn.execute(text("""
                                CREATE TABLE IF NOT EXISTS sessions (
                                    id VARCHAR PRIMARY KEY,
                                    user_id VARCHAR,
                                    thread_id VARCHAR,
                                    created_at TIMESTAMP,
                                    updated_at TIMESTAMP,
                                    last_activity TIMESTAMP,
                                    status VARCHAR,
                                    escalated BOOLEAN,
                                    escalation_ticket_id VARCHAR,
                                    metadata TEXT
                                )
                            """))
                            conn.commit()
                            logger.info("Sessions table created using raw SQL")
                    
                    if 'messages' in missing_tables:
                        with engine.connect() as conn:
                            conn.execute(text("""
                                CREATE TABLE IF NOT EXISTS messages (
                                    id VARCHAR PRIMARY KEY,
                                    session_id VARCHAR,
                                    role VARCHAR,
                                    content TEXT,
                                    timestamp TIMESTAMP,
                                    metadata TEXT,
                                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                                )
                            """))
                            conn.commit()
                            logger.info("Messages table created using raw SQL")
                    
                    # Final check
                    inspector = inspect(engine)
                    table_names = inspector.get_table_names()
                    missing_tables = [table for table in required_tables if table not in table_names]
                    
                    if missing_tables:
                        logger.error(f"Still missing tables after raw SQL creation: {missing_tables}")
                        raise Exception(f"Failed to create required tables: {missing_tables}")
                    else:
                        logger.info("All tables created successfully using raw SQL")
                        
                except Exception as e:
                    logger.error(f"Failed to create tables using raw SQL: {e}")
                    raise Exception(f"Failed to create required tables: {missing_tables}")
            else:
                logger.info("All tables created successfully after individual creation attempts")
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
        inspector = inspect(engine)
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
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        logger.info(f"Tables in database: {table_names}")
        
        # Check table schemas
        for table_name in table_names:
            columns = inspector.get_columns(table_name)
            logger.info(f"Table '{table_name}' columns: {[col['name'] for col in columns]}")
            
    except Exception as e:
        logger.error(f"Database debug error: {e}")
```

## Validation Steps

1. Replace the `database.py` file with the dynamic model discovery version
2. Restart the backend application
3. Check the logs to see which model classes are discovered
4. Verify that the tables are created using the discovered classes or raw SQL
5. Test the application functionality

## Additional Recommendations

1. **Model File Structure**: After this fix, consider examining the model files to understand their structure
2. **Model Registration**: Ensure that all models properly inherit from the Base declarative class
3. **Table Definitions**: Verify that the raw SQL table definitions match the model definitions
4. **Foreign Key Constraints**: Ensure that foreign key constraints are properly defined

This enhanced version should be able to handle the model class name issue by dynamically discovering the correct class names and using them for table creation. As a fallback, it will also attempt to create the missing tables using raw SQL, which should ensure that the application can start even if there are issues with the model definitions.

---

https://chat.z.ai/s/e0015b27-f0ee-4daf-9d27-cc8c6ed4e45c 

