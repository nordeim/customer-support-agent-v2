# Configuration Management - Meticulous Plan & Implementation

## 📋 Implementation Plan

### Overview
We'll create a comprehensive configuration management system that:
1. **Centralized Configuration** - Single source of truth for all settings
2. **Environment Support** - Development, staging, and production configurations
3. **Type Safety** - Pydantic validation with detailed error messages
4. **Security** - Proper handling of secrets and sensitive data
5. **Flexibility** - Easy override via environment variables
6. **Documentation** - Clear descriptions and examples for all settings

### Configuration Categories
- Application settings
- API configuration
- Database connections
- AI/ML model settings
- Caching and storage
- Security and authentication
- Monitoring and telemetry
- Rate limiting and performance
- File handling
- Feature flags

---

## 🛠️ Complete Implementation

### File 1: Comprehensive Configuration Module

**`backend/app/config.py`**
```python
"""
Comprehensive configuration management for Customer Support AI Agent.
Uses Pydantic Settings for type safety and validation.
"""
import os
import secrets
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from functools import lru_cache
from enum import Enum

from pydantic import BaseSettings, Field, validator, SecretStr, HttpUrl, PostgresDsn
from pydantic.types import conint, confloat


class Environment(str, Enum):
    """Application environment enumeration."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AIProvider(str, Enum):
    """AI provider enumeration."""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    LOCAL = "local"  # For testing without API


class Settings(BaseSettings):
    """
    Application settings with validation and type checking.
    All settings can be overridden via environment variables.
    """
    
    # ===========================
    # Application Settings
    # ===========================
    
    app_name: str = Field(
        default="Customer Support AI Agent",
        description="Application name used in logs and monitoring"
    )
    
    app_version: str = Field(
        default="1.0.0",
        description="Application version"
    )
    
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Current environment (development/staging/production)"
    )
    
    debug: bool = Field(
        default=False,
        description="Enable debug mode (verbose logging, debug endpoints)"
    )
    
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )
    
    # ===========================
    # API Configuration
    # ===========================
    
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    
    api_port: conint(ge=1, le=65535) = Field(
        default=8000,
        description="API server port"
    )
    
    api_prefix: str = Field(
        default="/api",
        description="API route prefix"
    )
    
    api_workers: conint(ge=1) = Field(
        default=1,
        description="Number of API worker processes"
    )
    
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins"
    )
    
    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS requests"
    )
    
    # ===========================
    # Security Configuration
    # ===========================
    
    secret_key: SecretStr = Field(
        default=None,
        description="Secret key for JWT and encryption (auto-generated if not set)"
    )
    
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )
    
    jwt_expiration_hours: conint(ge=1) = Field(
        default=24,
        description="JWT token expiration time in hours"
    )
    
    api_key_header: str = Field(
        default="X-API-Key",
        description="Header name for API key authentication"
    )
    
    encrypt_sessions: bool = Field(
        default=True,
        description="Enable session data encryption"
    )
    
    # ===========================
    # Database Configuration
    # ===========================
    
    database_url: str = Field(
        default="sqlite:///./data/customer_support.db",
        description="Primary database URL (SQLite or PostgreSQL)"
    )
    
    database_echo: bool = Field(
        default=False,
        description="Echo SQL statements (for debugging)"
    )
    
    database_pool_size: conint(ge=1) = Field(
        default=10,
        description="Database connection pool size"
    )
    
    database_pool_overflow: conint(ge=0) = Field(
        default=20,
        description="Maximum overflow connections"
    )
    
    database_pool_timeout: conint(ge=1) = Field(
        default=30,
        description="Connection pool timeout in seconds"
    )
    
    # ===========================
    # Redis Cache Configuration
    # ===========================
    
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    
    redis_password: Optional[SecretStr] = Field(
        default=None,
        description="Redis password (if required)"
    )
    
    redis_pool_size: conint(ge=1) = Field(
        default=10,
        description="Redis connection pool size"
    )
    
    redis_ttl: conint(ge=1) = Field(
        default=3600,
        description="Default cache TTL in seconds"
    )
    
    cache_enabled: bool = Field(
        default=True,
        description="Enable Redis caching"
    )
    
    # ===========================
    # ChromaDB Configuration
    # ===========================
    
    chroma_persist_directory: str = Field(
        default="./data/chroma_db",
        description="ChromaDB persistence directory"
    )
    
    chroma_collection_name: str = Field(
        default="customer_support_docs",
        description="ChromaDB collection name"
    )
    
    chroma_distance_function: str = Field(
        default="ip",
        description="Distance function for similarity (ip, l2, cosine)"
    )
    
    # ===========================
    # AI/ML Configuration
    # ===========================
    
    ai_provider: AIProvider = Field(
        default=AIProvider.OPENAI,
        description="AI provider to use"
    )
    
    # OpenAI Configuration
    openai_api_key: Optional[SecretStr] = Field(
        default=None,
        description="OpenAI API key"
    )
    
    openai_organization: Optional[str] = Field(
        default=None,
        description="OpenAI organization ID"
    )
    
    openai_api_base: Optional[HttpUrl] = Field(
        default=None,
        description="Custom OpenAI API base URL"
    )
    
    # Azure OpenAI Configuration
    azure_openai_endpoint: Optional[HttpUrl] = Field(
        default=None,
        description="Azure OpenAI endpoint URL"
    )
    
    azure_openai_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Azure OpenAI API key"
    )
    
    azure_openai_deployment: Optional[str] = Field(
        default=None,
        description="Azure OpenAI deployment name"
    )
    
    azure_openai_api_version: str = Field(
        default="2024-10-01-preview",
        description="Azure OpenAI API version"
    )
    
    # Model Configuration
    agent_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for agent responses"
    )
    
    agent_temperature: confloat(ge=0.0, le=2.0) = Field(
        default=0.7,
        description="Temperature for response generation"
    )
    
    agent_max_tokens: conint(ge=1) = Field(
        default=2000,
        description="Maximum tokens in agent response"
    )
    
    agent_timeout: conint(ge=1) = Field(
        default=30,
        description="Agent response timeout in seconds"
    )
    
    agent_max_retries: conint(ge=0) = Field(
        default=3,
        description="Maximum retries for failed API calls"
    )
    
    # Embedding Configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name (fallback if EmbeddingGemma unavailable)"
    )
    
    embedding_gemma_model: str = Field(
        default="google/embedding-gemma-256m-it",
        description="Google EmbeddingGemma model name"
    )
    
    embedding_dimension: conint(ge=1) = Field(
        default=768,
        description="Embedding vector dimension"
    )
    
    embedding_batch_size: conint(ge=1) = Field(
        default=32,
        description="Batch size for embedding generation"
    )
    
    use_gpu_embeddings: bool = Field(
        default=False,
        description="Use GPU for embedding generation"
    )
    
    # ===========================
    # RAG Configuration
    # ===========================
    
    rag_enabled: bool = Field(
        default=True,
        description="Enable RAG functionality"
    )
    
    rag_chunk_size: conint(ge=100) = Field(
        default=500,
        description="Text chunk size in words"
    )
    
    rag_chunk_overlap: conint(ge=0) = Field(
        default=50,
        description="Overlap between chunks in words"
    )
    
    rag_search_k: conint(ge=1) = Field(
        default=5,
        description="Default number of RAG results"
    )
    
    rag_similarity_threshold: confloat(ge=0.0, le=1.0) = Field(
        default=0.7,
        description="Minimum similarity score for RAG results"
    )
    
    rag_rerank_enabled: bool = Field(
        default=False,
        description="Enable result reranking"
    )
    
    # ===========================
    # Memory Configuration
    # ===========================
    
    memory_enabled: bool = Field(
        default=True,
        description="Enable conversation memory"
    )
    
    memory_max_entries: conint(ge=1) = Field(
        default=100,
        description="Maximum memory entries per session"
    )
    
    memory_ttl_hours: conint(ge=1) = Field(
        default=24,
        description="Memory TTL in hours"
    )
    
    memory_cleanup_days: conint(ge=1) = Field(
        default=30,
        description="Days before cleaning old memories"
    )
    
    # ===========================
    # File Handling Configuration
    # ===========================
    
    max_file_size: conint(ge=1) = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum file upload size in bytes"
    )
    
    allowed_file_types: List[str] = Field(
        default=[
            ".pdf", ".doc", ".docx", ".txt", ".md", 
            ".csv", ".xlsx", ".xls", ".json", ".xml",
            ".jpg", ".jpeg", ".png", ".gif", ".bmp",
            ".mp3", ".wav", ".m4a"
        ],
        description="Allowed file extensions for upload"
    )
    
    upload_directory: str = Field(
        default="./data/uploads",
        description="Directory for file uploads"
    )
    
    process_uploads_async: bool = Field(
        default=True,
        description="Process file uploads asynchronously"
    )
    
    # ===========================
    # Rate Limiting Configuration
    # ===========================
    
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    
    rate_limit_requests: conint(ge=1) = Field(
        default=100,
        description="Maximum requests per period"
    )
    
    rate_limit_period: conint(ge=1) = Field(
        default=60,
        description="Rate limit period in seconds"
    )
    
    rate_limit_burst: conint(ge=1) = Field(
        default=10,
        description="Burst allowance for rate limiting"
    )
    
    # ===========================
    # Monitoring & Telemetry
    # ===========================
    
    telemetry_enabled: bool = Field(
        default=True,
        description="Enable telemetry collection"
    )
    
    metrics_enabled: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    
    metrics_port: conint(ge=1, le=65535) = Field(
        default=9090,
        description="Prometheus metrics port"
    )
    
    otlp_endpoint: Optional[str] = Field(
        default=None,
        description="OpenTelemetry collector endpoint"
    )
    
    otlp_service_name: str = Field(
        default="customer-support-ai",
        description="Service name for telemetry"
    )
    
    sentry_dsn: Optional[SecretStr] = Field(
        default=None,
        description="Sentry DSN for error tracking"
    )
    
    # ===========================
    # WebSocket Configuration
    # ===========================
    
    websocket_enabled: bool = Field(
        default=True,
        description="Enable WebSocket support"
    )
    
    websocket_ping_interval: conint(ge=1) = Field(
        default=30,
        description="WebSocket ping interval in seconds"
    )
    
    websocket_ping_timeout: conint(ge=1) = Field(
        default=10,
        description="WebSocket ping timeout in seconds"
    )
    
    websocket_max_connections: conint(ge=1) = Field(
        default=1000,
        description="Maximum concurrent WebSocket connections"
    )
    
    # ===========================
    # Session Configuration
    # ===========================
    
    session_timeout_minutes: conint(ge=1) = Field(
        default=30,
        description="Session timeout in minutes"
    )
    
    session_max_messages: conint(ge=1) = Field(
        default=1000,
        description="Maximum messages per session"
    )
    
    session_cleanup_hours: conint(ge=1) = Field(
        default=24,
        description="Hours before cleaning inactive sessions"
    )
    
    # ===========================
    # Escalation Configuration
    # ===========================
    
    escalation_enabled: bool = Field(
        default=True,
        description="Enable escalation to human support"
    )
    
    escalation_confidence_threshold: confloat(ge=0.0, le=1.0) = Field(
        default=0.7,
        description="Confidence threshold for escalation"
    )
    
    escalation_keywords: List[str] = Field(
        default=["urgent", "emergency", "legal", "complaint", "manager"],
        description="Keywords that trigger escalation"
    )
    
    escalation_notification_email: Optional[str] = Field(
        default=None,
        description="Email for escalation notifications"
    )
    
    escalation_webhook_url: Optional[HttpUrl] = Field(
        default=None,
        description="Webhook URL for escalation notifications"
    )
    
    # ===========================
    # Feature Flags
    # ===========================
    
    feature_voice_input: bool = Field(
        default=False,
        description="Enable voice input feature"
    )
    
    feature_multilingual: bool = Field(
        default=False,
        description="Enable multilingual support"
    )
    
    feature_analytics: bool = Field(
        default=False,
        description="Enable analytics tracking"
    )
    
    feature_export_chat: bool = Field(
        default=True,
        description="Enable chat export feature"
    )
    
    feature_file_preview: bool = Field(
        default=True,
        description="Enable file preview feature"
    )
    
    # ===========================
    # Development Settings
    # ===========================
    
    dev_auto_reload: bool = Field(
        default=False,
        description="Enable auto-reload in development"
    )
    
    dev_sample_data: bool = Field(
        default=False,
        description="Load sample data on startup"
    )
    
    dev_mock_ai: bool = Field(
        default=False,
        description="Use mock AI responses (no API calls)"
    )
    
    dev_slow_mode: bool = Field(
        default=False,
        description="Simulate slow responses for testing"
    )
    
    # ===========================
    # Validators
    # ===========================
    
    @validator('secret_key', pre=True, always=True)
    def generate_secret_key(cls, v):
        """Generate a secret key if not provided."""
        if not v:
            return SecretStr(secrets.token_urlsafe(32))
        return v
    
    @validator('environment', pre=True)
    def validate_environment(cls, v):
        """Validate and normalize environment."""
        if isinstance(v, str):
            v = v.lower()
        return v
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @validator('database_url')
    def validate_database_url(cls, v, values):
        """Validate database URL and create directories for SQLite."""
        if v.startswith('sqlite:///'):
            # Extract path and create directory if needed
            db_path = v.replace('sqlite:///', '')
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.isabs(db_path):
                Path(db_dir).mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('upload_directory', 'chroma_persist_directory')
    def create_directories(cls, v):
        """Create directories if they don't exist."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('debug', pre=True, always=True)
    def set_debug_from_environment(cls, v, values):
        """Set debug based on environment if not explicitly set."""
        if v is None:
            env = values.get('environment')
            return env == Environment.DEVELOPMENT
        return v
    
    @validator('api_workers', pre=True, always=True)
    def set_workers_from_environment(cls, v, values):
        """Set worker count based on environment."""
        if v == 1:
            env = values.get('environment')
            if env == Environment.PRODUCTION:
                # Use CPU count in production
                return min(os.cpu_count() or 1, 4)
        return v
    
    @validator('log_level', pre=True, always=True)
    def set_log_level_from_debug(cls, v, values):
        """Set log level based on debug mode."""
        if values.get('debug'):
            return LogLevel.DEBUG
        return v
    
    # ===========================
    # Computed Properties
    # ===========================
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def redis_url_with_password(self) -> str:
        """Get Redis URL with password if configured."""
        if self.redis_password:
            # Parse URL and insert password
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(self.redis_url)
            netloc = f":{self.redis_password.get_secret_value()}@{parsed.hostname}:{parsed.port}"
            return urlunparse(parsed._replace(netloc=netloc))
        return self.redis_url
    
    @property
    def requires_ai_credentials(self) -> bool:
        """Check if AI credentials are configured."""
        if self.ai_provider == AIProvider.OPENAI:
            return bool(self.openai_api_key)
        elif self.ai_provider == AIProvider.AZURE_OPENAI:
            return bool(self.azure_openai_api_key and self.azure_openai_endpoint)
        return False
    
    @property
    def database_is_sqlite(self) -> bool:
        """Check if using SQLite database."""
        return self.database_url.startswith('sqlite')
    
    @property
    def database_is_postgres(self) -> bool:
        """Check if using PostgreSQL database."""
        return self.database_url.startswith('postgresql')
    
    # ===========================
    # Methods
    # ===========================
    
    def validate_configuration(self) -> List[str]:
        """
        Validate configuration and return any warnings.
        
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Check production requirements
        if self.is_production:
            if self.debug:
                warnings.append("Debug mode is enabled in production")
            
            if self.database_is_sqlite:
                warnings.append("SQLite is not recommended for production")
            
            if not self.secret_key or len(self.secret_key.get_secret_value()) < 32:
                warnings.append("Secret key should be at least 32 characters in production")
            
            if not self.telemetry_enabled:
                warnings.append("Telemetry is disabled in production")
            
            if self.cors_origins == ["http://localhost:3000"]:
                warnings.append("CORS origins still set to localhost in production")
        
        # Check AI configuration
        if not self.requires_ai_credentials and not self.dev_mock_ai:
            warnings.append("No AI credentials configured and mock mode is disabled")
        
        # Check Redis configuration
        if self.cache_enabled and "localhost" in self.redis_url and self.is_production:
            warnings.append("Redis is configured with localhost in production")
        
        # Check file upload directory
        if not Path(self.upload_directory).exists():
            warnings.append(f"Upload directory does not exist: {self.upload_directory}")
        
        return warnings
    
    def get_safe_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary with secrets masked.
        
        Returns:
            Dictionary with configuration values (secrets masked)
        """
        config_dict = self.dict()
        
        # Mask sensitive fields
        sensitive_fields = [
            'secret_key', 'openai_api_key', 'azure_openai_api_key',
            'redis_password', 'sentry_dsn', 'database_url'
        ]
        
        for field in sensitive_fields:
            if field in config_dict and config_dict[field]:
                if isinstance(config_dict[field], SecretStr):
                    config_dict[field] = "***MASKED***"
                elif isinstance(config_dict[field], str) and len(config_dict[field]) > 0:
                    # Mask database URLs partially
                    if 'url' in field.lower():
                        # Show only the scheme and host
                        from urllib.parse import urlparse
                        parsed = urlparse(config_dict[field])
                        config_dict[field] = f"{parsed.scheme}://***MASKED***"
                    else:
                        config_dict[field] = "***MASKED***"
        
        return config_dict
    
    def write_env_template(self, filepath: str = ".env.example") -> None:
        """
        Write an example .env file with all settings.
        
        Args:
            filepath: Path to write the example file
        """
        with open(filepath, 'w') as f:
            f.write("# Customer Support AI Agent Configuration\n")
            f.write("# Copy this file to .env and update with your values\n\n")
            
            for field_name, field in self.__fields__.items():
                # Write description as comment
                if field.field_info.description:
                    f.write(f"# {field.field_info.description}\n")
                
                # Write the field with example value
                value = getattr(self, field_name)
                if isinstance(value, SecretStr):
                    value = "your-secret-here"
                elif isinstance(value, Enum):
                    value = value.value
                elif isinstance(value, list):
                    value = ",".join(str(v) for v in value)
                
                env_name = field_name.upper()
                f.write(f"# {env_name}={value}\n\n")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Allow extra fields in development
        extra = "ignore"


# ===========================
# Settings Management
# ===========================

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings instance
    """
    settings = Settings()
    
    # Log configuration warnings
    warnings = settings.validate_configuration()
    if warnings:
        import logging
        logger = logging.getLogger(__name__)
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")
    
    return settings


# Create global settings instance
settings = get_settings()

# Export for convenience
__all__ = ['Settings', 'get_settings', 'settings', 'Environment', 'LogLevel', 'AIProvider']
```

### File 2: Environment Variables Template

**`.env.example`**
```env
# ==============================================================================
# Customer Support AI Agent - Environment Configuration
# ==============================================================================
# Copy this file to .env and update with your actual values
# DO NOT commit .env file to version control
# ==============================================================================

# -----------------------------------------------------------------------------
# Application Settings
# -----------------------------------------------------------------------------

# Application name (used in logs and monitoring)
APP_NAME="Customer Support AI Agent"

# Application version
APP_VERSION="1.0.0"

# Environment: development, testing, staging, production
ENVIRONMENT="development"

# Enable debug mode (verbose logging, debug endpoints)
# Automatically set to true in development, false in production
DEBUG="false"

# Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL="INFO"

# -----------------------------------------------------------------------------
# API Configuration
# -----------------------------------------------------------------------------

# API server host (0.0.0.0 for all interfaces)
API_HOST="0.0.0.0"

# API server port
API_PORT="8000"

# API route prefix
API_PREFIX="/api"

# Number of worker processes (1 for development, auto-scaled in production)
API_WORKERS="1"

# CORS allowed origins (comma-separated)
CORS_ORIGINS="http://localhost:3000,http://localhost:3001"

# Allow credentials in CORS requests
CORS_ALLOW_CREDENTIALS="true"

# -----------------------------------------------------------------------------
# Security Configuration
# -----------------------------------------------------------------------------

# Secret key for JWT and encryption (auto-generated if not set)
# IMPORTANT: Set a strong secret key in production!
# Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
SECRET_KEY=""

# JWT configuration
JWT_ALGORITHM="HS256"
JWT_EXPIRATION_HOURS="24"

# API key header name
API_KEY_HEADER="X-API-Key"

# Enable session data encryption
ENCRYPT_SESSIONS="true"

# -----------------------------------------------------------------------------
# Database Configuration
# -----------------------------------------------------------------------------

# Primary database URL
# SQLite (development): sqlite:///./data/customer_support.db
# PostgreSQL (production): postgresql://user:password@localhost:5432/dbname
DATABASE_URL="sqlite:///./data/customer_support.db"

# Database settings
DATABASE_ECHO="false"
DATABASE_POOL_SIZE="10"
DATABASE_POOL_OVERFLOW="20"
DATABASE_POOL_TIMEOUT="30"

# -----------------------------------------------------------------------------
# Redis Cache Configuration
# -----------------------------------------------------------------------------

# Redis connection URL
REDIS_URL="redis://localhost:6379/0"

# Redis password (if required)
# REDIS_PASSWORD=""

# Redis settings
REDIS_POOL_SIZE="10"
REDIS_TTL="3600"
CACHE_ENABLED="true"

# -----------------------------------------------------------------------------
# ChromaDB Vector Store Configuration
# -----------------------------------------------------------------------------

# ChromaDB persistence directory
CHROMA_PERSIST_DIRECTORY="./data/chroma_db"

# ChromaDB collection name
CHROMA_COLLECTION_NAME="customer_support_docs"

# Distance function: ip (inner product), l2, cosine
CHROMA_DISTANCE_FUNCTION="ip"

# -----------------------------------------------------------------------------
# AI/ML Configuration
# -----------------------------------------------------------------------------

# AI provider: openai, azure_openai, local
AI_PROVIDER="openai"

# === OpenAI Configuration ===
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=""

# OpenAI organization ID (optional)
# OPENAI_ORGANIZATION=""

# Custom OpenAI API base URL (optional, for proxies)
# OPENAI_API_BASE=""

# === Azure OpenAI Configuration ===
# Azure OpenAI endpoint (required if using Azure)
# AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"

# Azure OpenAI API key
# AZURE_OPENAI_API_KEY=""

# Azure OpenAI deployment name
# AZURE_OPENAI_DEPLOYMENT="your-deployment"

# Azure OpenAI API version
AZURE_OPENAI_API_VERSION="2024-10-01-preview"

# === Model Configuration ===

# LLM model for agent responses
AGENT_MODEL="gpt-4o-mini"

# Temperature for response generation (0.0-2.0)
AGENT_TEMPERATURE="0.7"

# Maximum tokens in agent response
AGENT_MAX_TOKENS="2000"

# Agent response timeout in seconds
AGENT_TIMEOUT="30"

# Maximum retries for failed API calls
AGENT_MAX_RETRIES="3"

# === Embedding Configuration ===

# Embedding model (fallback if EmbeddingGemma unavailable)
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# Google EmbeddingGemma model
EMBEDDING_GEMMA_MODEL="google/embedding-gemma-256m-it"

# Embedding vector dimension
EMBEDDING_DIMENSION="768"

# Batch size for embedding generation
EMBEDDING_BATCH_SIZE="32"

# Use GPU for embeddings (requires CUDA)
USE_GPU_EMBEDDINGS="false"

# -----------------------------------------------------------------------------
# RAG (Retrieval-Augmented Generation) Configuration
# -----------------------------------------------------------------------------

# Enable RAG functionality
RAG_ENABLED="true"

# Chunking settings
RAG_CHUNK_SIZE="500"
RAG_CHUNK_OVERLAP="50"

# Search settings
RAG_SEARCH_K="5"
RAG_SIMILARITY_THRESHOLD="0.7"
RAG_RERANK_ENABLED="false"

# -----------------------------------------------------------------------------
# Memory Configuration
# -----------------------------------------------------------------------------

# Enable conversation memory
MEMORY_ENABLED="true"

# Memory limits
MEMORY_MAX_ENTRIES="100"
MEMORY_TTL_HOURS="24"
MEMORY_CLEANUP_DAYS="30"

# -----------------------------------------------------------------------------
# File Handling Configuration
# -----------------------------------------------------------------------------

# Maximum file upload size in bytes (10MB)
MAX_FILE_SIZE="10485760"

# Allowed file extensions (comma-separated)
ALLOWED_FILE_TYPES=".pdf,.doc,.docx,.txt,.md,.csv,.xlsx,.xls,.json,.xml,.jpg,.jpeg,.png,.gif,.bmp,.mp3,.wav,.m4a"

# Upload directory
UPLOAD_DIRECTORY="./data/uploads"

# Process uploads asynchronously
PROCESS_UPLOADS_ASYNC="true"

# -----------------------------------------------------------------------------
# Rate Limiting Configuration
# -----------------------------------------------------------------------------

# Enable rate limiting
RATE_LIMIT_ENABLED="true"

# Rate limit settings
RATE_LIMIT_REQUESTS="100"
RATE_LIMIT_PERIOD="60"
RATE_LIMIT_BURST="10"

# -----------------------------------------------------------------------------
# Monitoring & Telemetry
# -----------------------------------------------------------------------------

# Enable telemetry collection
TELEMETRY_ENABLED="true"

# Enable Prometheus metrics
METRICS_ENABLED="true"
METRICS_PORT="9090"

# OpenTelemetry collector endpoint (optional)
# OTLP_ENDPOINT="http://localhost:4317"
OTLP_SERVICE_NAME="customer-support-ai"

# Sentry error tracking DSN (optional)
# SENTRY_DSN="https://your-key@sentry.io/project-id"

# -----------------------------------------------------------------------------
# WebSocket Configuration
# -----------------------------------------------------------------------------

# Enable WebSocket support
WEBSOCKET_ENABLED="true"

# WebSocket settings
WEBSOCKET_PING_INTERVAL="30"
WEBSOCKET_PING_TIMEOUT="10"
WEBSOCKET_MAX_CONNECTIONS="1000"

# -----------------------------------------------------------------------------
# Session Configuration
# -----------------------------------------------------------------------------

# Session settings
SESSION_TIMEOUT_MINUTES="30"
SESSION_MAX_MESSAGES="1000"
SESSION_CLEANUP_HOURS="24"

# -----------------------------------------------------------------------------
# Escalation Configuration
# -----------------------------------------------------------------------------

# Enable escalation to human support
ESCALATION_ENABLED="true"

# Escalation settings
ESCALATION_CONFIDENCE_THRESHOLD="0.7"
ESCALATION_KEYWORDS="urgent,emergency,legal,complaint,manager"

# Escalation notifications (optional)
# ESCALATION_NOTIFICATION_EMAIL="support@example.com"
# ESCALATION_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# -----------------------------------------------------------------------------
# Feature Flags
# -----------------------------------------------------------------------------

# Enable/disable features
FEATURE_VOICE_INPUT="false"
FEATURE_MULTILINGUAL="false"
FEATURE_ANALYTICS="false"
FEATURE_EXPORT_CHAT="true"
FEATURE_FILE_PREVIEW="true"

# -----------------------------------------------------------------------------
# Development Settings (Only for development/testing)
# -----------------------------------------------------------------------------

# Development features
DEV_AUTO_RELOAD="false"
DEV_SAMPLE_DATA="false"
DEV_MOCK_AI="false"
DEV_SLOW_MODE="false"

# ==============================================================================
# End of Configuration
# ==============================================================================
```

### File 3: Comprehensive .gitignore

**`.gitignore`**
```gitignore
# ==============================================================================
# Customer Support AI Agent - Git Ignore File
# ==============================================================================

# -----------------------------------------------------------------------------
# Environment and Secrets
# -----------------------------------------------------------------------------
.env
.env.*
!.env.example
*.key
*.pem
*.cert
*.crt
secrets/
credentials/

# -----------------------------------------------------------------------------
# Python
# -----------------------------------------------------------------------------
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
ENV/
env/
.venv/
.virtualenv/
backend/venv/
backend/.venv/

# PyInstaller
*.manifest
*.spec

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# pyenv
.python-version

# Celery
celerybeat-schedule
celerybeat.pid

# MyPy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# -----------------------------------------------------------------------------
# Node.js / Frontend
# -----------------------------------------------------------------------------
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
lerna-debug.log*
.pnpm-debug.log*
package-lock.json
yarn.lock
pnpm-lock.yaml

# Build outputs
frontend/dist/
frontend/build/
.next/
out/
.nuxt/
.cache/
.parcel-cache/

# Testing
frontend/coverage/
.nyc_output/

# Production builds
*.production
*.production.local

# -----------------------------------------------------------------------------
# Databases and Storage
# -----------------------------------------------------------------------------
# SQLite
*.db
*.sqlite
*.sqlite3
*.db-journal
*.db-wal
*.db-shm
backend/data/*.db
data/*.db

# PostgreSQL
*.sql
*.dump
*.backup

# ChromaDB / Vector stores
chroma_db/
backend/data/chroma_db/
chromadb/
*.index
*.bin

# Redis
dump.rdb
redis.conf.local

# File uploads
uploads/
backend/data/uploads/
backend/uploads/
temp/
tmp/

# -----------------------------------------------------------------------------
# Logs and Monitoring
# -----------------------------------------------------------------------------
logs/
*.log
*.log.*
backend/logs/
frontend/logs/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
pnpm-debug.log*

# -----------------------------------------------------------------------------
# IDE and Editors
# -----------------------------------------------------------------------------
# Visual Studio Code
.vscode/
*.code-workspace
.history/

# JetBrains IDEs
.idea/
*.iml
*.iws
*.ipr
.idea_modules/

# Sublime Text
*.sublime-project
*.sublime-workspace

# Vim
*.swp
*.swo
*.swn
.vim/
Session.vim

# Emacs
*~
\#*\#
.\#*
.emacs.desktop
.emacs.desktop.lock

# MacOS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
.AppleDouble
.LSOverride

# Windows
Desktop.ini
$RECYCLE.BIN/
*.lnk

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------
# Docker volumes (local development)
docker/volumes/
.docker/

# Docker override files
docker-compose.override.yml
docker-compose.*.yml
!docker-compose.prod.yml
!docker-compose.dev.yml

# -----------------------------------------------------------------------------
# Testing and Quality
# -----------------------------------------------------------------------------
# Test outputs
test-results/
test-report.xml
junit.xml
.benchmarks/

# Code coverage
coverage/
.coverage
htmlcov/
lcov.info
*.gcov

# Load testing
artillery_report*.json
k6_results/

# -----------------------------------------------------------------------------
# Build and Compilation
# -----------------------------------------------------------------------------
# General build outputs
build/
dist/
out/
target/
bin/
obj/

# Webpack
.webpack/
webpack-stats.json

# Vite
frontend/.vite/

# -----------------------------------------------------------------------------
# Operating System Files
# -----------------------------------------------------------------------------
# Temporary files
*.tmp
*.temp
*.cache
~$*

# Backup files
*.bak
*.backup
*.old
.backup/

# -----------------------------------------------------------------------------
# Project Specific
# -----------------------------------------------------------------------------
# Data directories (but keep structure)
backend/data/*
!backend/data/.gitkeep
data/*
!data/.gitkeep

# Model files
models/*.bin
models/*.pt
models/*.pth
models/*.onnx
models/*.pb
*.safetensors

# Configuration overrides
config.local.yml
settings.local.json
*.local.config

# Monitoring data
prometheus_data/
grafana_data/
metrics/

# Documentation builds
docs/_build/
docs/.doctrees/
site/

# Certificates (development)
certs/
ssl/

# Seeds and fixtures (may contain sensitive data)
seeds/*.json
fixtures/*.json

# -----------------------------------------------------------------------------
# Miscellaneous
# -----------------------------------------------------------------------------
# Archives
*.zip
*.tar.gz
*.rar
*.7z

# Terraform
*.tfstate
*.tfstate.*
.terraform/

# Kubernetes
kubeconfig
*.kubeconfig

# Other
.env.local
.env.development.local
.env.test.local
.env.production.local
.notes
TODO.md
NOTES.md
.scratch/

# ==============================================================================
# End of .gitignore
# ==============================================================================
```

## 📝 Summary

### ✅ Complete Configuration Management System Delivered

I've created a comprehensive configuration management system with:

#### **1. Advanced Configuration Module** (`config.py`)
- **90+ Configuration Settings** covering all aspects of the application
- **Type Safety** with Pydantic validation and custom validators
- **Environment Support** for development, staging, and production
- **Smart Defaults** with automatic adjustments based on environment
- **Validation Methods** to check configuration integrity
- **Security Features** like automatic secret generation and masking
- **Computed Properties** for derived configuration values
- **Helper Methods** for configuration management and debugging

#### **2. Comprehensive Environment Template** (`.env.example`)
- **Complete Documentation** for every configuration option
- **Organized by Category** for easy navigation
- **Example Values** that are safe to commit
- **Security Guidelines** for sensitive settings
- **Setup Instructions** for different environments
- **Feature Flags** for progressive rollout

#### **3. Professional .gitignore**
- **Language-Specific** patterns for Python and Node.js
- **IDE Coverage** for all popular editors
- **Security Focus** excluding all sensitive files
- **Database Files** for SQLite, PostgreSQL, and vector stores
- **Build Artifacts** and temporary files
- **Platform-Specific** patterns for Windows, macOS, and Linux
- **Docker & Kubernetes** configuration exclusions

### 🎯 Key Features

1. **Type-Safe Configuration**: Every setting is validated with Pydantic
2. **Environment-Aware**: Automatic adjustments based on deployment environment
3. **Security-First**: Secrets are protected and masked in logs
4. **Validation**: Built-in configuration validation with helpful warnings
5. **Flexibility**: Easy override via environment variables
6. **Documentation**: Every setting is documented with descriptions
7. **Developer-Friendly**: Helpful defaults for development
8. **Production-Ready**: Strict validation for production deployments

### 🚀 Usage Examples

```python
# Import settings
from app.config import settings

# Use in application
if settings.is_production:
    # Production-specific code
    pass

# Check configuration
warnings = settings.validate_configuration()
if warnings:
    for warning in warnings:
        logger.warning(warning)

# Get safe configuration for logging
safe_config = settings.get_safe_dict()
logger.info(f"Configuration: {safe_config}")

# Generate .env template
settings.write_env_template(".env.example")
```

### 🔒 Security Best Practices

1. **Never commit `.env`** - Only `.env.example` should be in version control
2. **Use strong secrets** - Generate with provided command
3. **Mask sensitive data** - Automatic masking in logs
4. **Validate in production** - Built-in production checks
5. **Separate environments** - Different configs for dev/staging/prod

The configuration system is now complete, secure, and ready for deployment across all environments!
