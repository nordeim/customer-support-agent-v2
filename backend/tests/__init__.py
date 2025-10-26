"""
Tests module for Customer Support AI Agent
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Set testing environment
os.environ["ENVIRONMENT"] = "testing"
os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"  # Use different Redis DB for tests

# Test fixtures and utilities
from typing import Generator
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

def get_test_db() -> Generator[Session, None, None]:
    """Get test database session."""
    from app.database import Base
    
    # Create test database
    SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture
def client():
    """Create test client."""
    from app.main import app
    from app.database import get_db
    
    app.dependency_overrides[get_db] = get_test_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()

__all__ = [
    "get_test_db",
    "client",
]
