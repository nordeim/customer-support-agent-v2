"""
Customer Support AI Agent Backend Application
"""

__version__ = "1.0.0"
__author__ = "Customer Support AI Team"

# Application metadata
APP_NAME = "Customer Support AI Agent"
APP_DESCRIPTION = "AI-powered customer support system with RAG and conversation memory"

# Import key components for easier access
from .config import settings, get_settings
from .database import get_db, init_db
from .main import app

__all__ = [
    "app",
    "settings",
    "get_settings",
    "get_db",
    "init_db",
    "APP_NAME",
    "APP_DESCRIPTION",
    "__version__",
]
