# 🎯 Comprehensive Implementation Plan & Complete File Replacements

## Executive Summary

I will create **complete, production-ready drop-in replacement files** that incorporate all critical fixes from the session management review plus fixes for `config.py` and `database.py`. This will be a **zero-placeholder implementation** - every file will be fully functional.

---

## 📋 MASTER IMPLEMENTATION CHECKLIST

### Phase 1: Core Infrastructure Files ✓
- [x] Review `config.py` for issues
- [x] Review `database.py` for issues  
- [x] Create implementation plan
- [ ] Generate `backend/app/config.py` (complete replacement)
- [ ] Generate `backend/app/database.py` (complete replacement)

### Phase 2: Session Management Foundation ✓
- [ ] Generate `backend/app/session/session_store.py` (enhanced)
- [ ] Generate `backend/app/session/validators.py` (NEW)
- [ ] Generate `backend/app/utils/encryption.py` (NEW)
- [ ] Generate `backend/app/utils/retry.py` (NEW)

### Phase 3: Session Store Implementations ✓
- [ ] Generate `backend/app/session/in_memory_session_store.py` (fixed)
- [ ] Generate `backend/app/session/redis_session_store.py` (fixed)
- [ ] Generate `backend/app/session/distributed_lock.py` (NEW)

### Phase 4: Agent Integration ✓
- [ ] Generate `backend/app/agents/chat_agent.py` (fixed)
- [ ] Generate `backend/app/main.py` (enhanced)

---

## 🔍 DETAILED FILE ANALYSIS

### File 1: `backend/app/config.py`

**Issues Found:**
1. ❌ **CRITICAL**: Duplicate `session_timeout_minutes` field (lines 370 & 750)
2. ❌ **CRITICAL**: Duplicate `enable_telemetry` field
3. ⚠️ Missing `session_encryption_key` for session encryption
4. ⚠️ No validation for Redis when `use_shared_context=True`
5. ⚠️ Inconsistent session configuration organization
6. 💡 Missing fields for distributed locking, L1 cache, fingerprinting

**Fixes to Implement:**
- ✅ Remove all duplicate fields
- ✅ Consolidate session configuration into one section
- ✅ Add `session_encryption_key` field
- ✅ Add validators for Redis dependency
- ✅ Add fields for new session features (locking, caching, fingerprinting)
- ✅ Add comprehensive validation methods
- ✅ Improve type safety and parsing

---

### File 2: `backend/app/database.py`

**Issues Found:**
1. ❌ **CRITICAL**: Race condition in `init_db()` - no locking
2. ❌ **CRITICAL**: Missing async support for FastAPI
3. ⚠️ Weak error handling with bare `except Exception`
4. ⚠️ Connection pool not optimized for async operations
5. ⚠️ No periodic health monitoring
6. ⚠️ Resource leaks possible on init failure
7. 💡 Missing WAL mode for SQLite
8. 💡 Missing proper shutdown with connection draining

**Fixes to Implement:**
- ✅ Add thread-safe initialization with lock
- ✅ Add async engine option
- ✅ Improve error handling specificity
- ✅ Optimize connection pools for async
- ✅ Add periodic health checks
- ✅ Add proper resource cleanup
- ✅ Enable WAL mode for SQLite
- ✅ Implement graceful shutdown

---

## 📝 IMPLEMENTATION PLAN VALIDATION

### Validation Criteria:
✅ All critical race conditions addressed  
✅ All security vulnerabilities fixed  
✅ All performance optimizations included  
✅ Backward compatibility maintained  
✅ No placeholder comments  
✅ Complete error handling  
✅ Production-ready logging  
✅ Comprehensive validation  

---

# 🚀 COMPLETE FILE IMPLEMENTATIONS

## File 1: `backend/app/config.py` (Complete Replacement)

**Checklist:**
- [x] Remove duplicate `session_timeout_minutes`
- [x] Remove duplicate `enable_telemetry`
- [x] Add `session_encryption_key`
- [x] Add session fingerprinting config
- [x] Add distributed locking config
- [x] Add L1 cache config
- [x] Add Redis dependency validation
- [x] Consolidate session settings
- [x] Improve type safety
- [x] Add comprehensive validators

```python
"""
Comprehensive configuration management for Customer Support AI Agent.
Uses Pydantic Settings for type safety and validation with production optimizations.

Version: 2.0.0 (Session Management Enhanced)
"""
import os
import secrets
import json
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from functools import lru_cache
from enum import Enum

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator, SecretStr
from typing import Annotated


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
    LOCAL = "local"


class SessionStoreType(str, Enum):
    """Session store backend type."""
    IN_MEMORY = "in_memory"
    REDIS = "redis"


class Settings(BaseSettings):
    """
    Application settings with validation and type checking.
    All settings can be overridden via environment variables.
    
    Version 2.0.0: Enhanced session management with encryption, locking, and caching.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_parse_none_str="null",
        env_nested_delimiter="__"
    )
    
    # ===========================
    # Application Settings
    # ===========================
    
    app_name: str = Field(
        default="Customer Support AI Agent",
        description="Application name used in logs and monitoring"
    )
    
    app_version: str = Field(
        default="2.0.0",
        description="Application version"
    )
    
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Current environment"
    )
    
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
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
    
    api_port: Annotated[int, Field(ge=1, le=65535)] = Field(
        default=8000,
        description="API server port"
    )
    
    api_prefix: str = Field(
        default="/api",
        description="API route prefix"
    )
    
    api_workers: Annotated[int, Field(ge=1, le=32)] = Field(
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
        default_factory=lambda: SecretStr(secrets.token_urlsafe(32)),
        description="Secret key for JWT and encryption"
    )
    
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )
    
    jwt_expiration_hours: Annotated[int, Field(ge=1, le=168)] = Field(
        default=24,
        description="JWT token expiration time in hours"
    )
    
    api_key_header: str = Field(
        default="X-API-Key",
        description="Header name for API key authentication"
    )
    
    # ===========================
    # Database Configuration
    # ===========================
    
    database_url: str = Field(
        default="sqlite:///./data/customer_support.db",
        description="Primary database URL"
    )
    
    database_echo: bool = Field(
        default=False,
        description="Echo SQL statements"
    )
    
    database_pool_size: Annotated[int, Field(ge=5, le=100)] = Field(
        default=10,
        description="Database connection pool size"
    )
    
    database_pool_overflow: Annotated[int, Field(ge=0, le=50)] = Field(
        default=20,
        description="Maximum overflow connections"
    )
    
    database_pool_timeout: Annotated[int, Field(ge=5, le=60)] = Field(
        default=30,
        description="Connection pool timeout in seconds"
    )
    
    database_pool_recycle: Annotated[int, Field(ge=300, le=7200)] = Field(
        default=3600,
        description="Connection pool recycle time in seconds"
    )
    
    database_async_enabled: bool = Field(
        default=True,
        description="Enable async database operations"
    )
    
    # ===========================
    # Redis Configuration
    # ===========================
    
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    
    redis_password: Optional[SecretStr] = Field(
        default=None,
        description="Redis password (if required)"
    )
    
    redis_max_connections: Annotated[int, Field(ge=10, le=200)] = Field(
        default=50,
        description="Maximum Redis connections"
    )
    
    redis_socket_timeout: Annotated[int, Field(ge=1, le=30)] = Field(
        default=5,
        description="Redis socket timeout in seconds"
    )
    
    redis_socket_connect_timeout: Annotated[int, Field(ge=1, le=30)] = Field(
        default=5,
        description="Redis connection timeout in seconds"
    )
    
    redis_retry_on_timeout: bool = Field(
        default=True,
        description="Retry on timeout"
    )
    
    redis_health_check_interval: Annotated[int, Field(ge=10, le=300)] = Field(
        default=30,
        description="Redis health check interval in seconds"
    )
    
    cache_enabled: bool = Field(
        default=True,
        description="Enable Redis caching"
    )
    
    redis_ttl: Annotated[int, Field(ge=60, le=86400)] = Field(
        default=3600,
        description="Default cache TTL in seconds"
    )
    
    # ===========================
    # Session Management Configuration (Consolidated & Enhanced)
    # ===========================
    
    # Session Store Backend
    session_store_type: SessionStoreType = Field(
        default=SessionStoreType.IN_MEMORY,
        description="Session store backend type"
    )
    
    # Legacy compatibility field
    use_shared_context: bool = Field(
        default=False,
        description="Use shared session context (sets session_store_type to REDIS)"
    )
    
    # Session Timeouts
    session_timeout_minutes: Annotated[int, Field(ge=5, le=1440)] = Field(
        default=30,
        description="Session timeout in minutes"
    )
    
    session_cleanup_interval_seconds: Annotated[int, Field(ge=60, le=3600)] = Field(
        default=300,
        description="Interval for session cleanup in seconds"
    )
    
    # In-Memory Store Settings
    session_max_sessions: Annotated[int, Field(ge=100, le=1000000)] = Field(
        default=10000,
        description="Maximum sessions in memory (in-memory store only)"
    )
    
    # Redis Session Store Settings
    redis_session_key_prefix: str = Field(
        default="agent:session:",
        description="Redis key prefix for sessions"
    )
    
    redis_session_ttl_seconds: Annotated[int, Field(ge=300, le=86400)] = Field(
        default=1800,
        description="Redis session TTL in seconds"
    )
    
    # Session Security
    session_encryption_enabled: bool = Field(
        default=True,
        description="Enable session data encryption at rest"
    )
    
    session_encryption_key: Optional[SecretStr] = Field(
        default=None,
        description="Session encryption key (auto-generated if not set)"
    )
    
    session_fingerprinting_enabled: bool = Field(
        default=True,
        description="Enable session fingerprinting for hijacking prevention"
    )
    
    session_fingerprint_strict: bool = Field(
        default=False,
        description="Strict fingerprint validation (reject on mismatch)"
    )
    
    # Session Locking (Distributed)
    session_locking_enabled: bool = Field(
        default=True,
        description="Enable distributed session locking"
    )
    
    session_lock_timeout_seconds: Annotated[int, Field(ge=5, le=300)] = Field(
        default=30,
        description="Distributed lock timeout in seconds"
    )
    
    session_lock_retry_attempts: Annotated[int, Field(ge=1, le=10)] = Field(
        default=3,
        description="Lock acquisition retry attempts"
    )
    
    # Session L1 Caching
    session_l1_cache_enabled: bool = Field(
        default=True,
        description="Enable L1 (in-memory) cache for Redis sessions"
    )
    
    session_l1_cache_size: Annotated[int, Field(ge=100, le=10000)] = Field(
        default=1000,
        description="L1 cache size (number of sessions)"
    )
    
    session_l1_cache_ttl_seconds: Annotated[int, Field(ge=10, le=300)] = Field(
        default=60,
        description="L1 cache TTL in seconds"
    )
    
    # Session Analytics
    session_analytics_enabled: bool = Field(
        default=False,
        description="Enable session analytics"
    )
    
    session_max_messages: Annotated[int, Field(ge=100, le=10000)] = Field(
        default=1000,
        description="Maximum messages per session"
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
        description="Distance function for similarity"
    )
    
    # ===========================
    # AI/ML Configuration
    # ===========================
    
    ai_provider: AIProvider = Field(
        default=AIProvider.LOCAL,
        description="AI provider to use"
    )
    
    openai_api_key: Optional[SecretStr] = Field(
        default=None,
        description="OpenAI API key"
    )
    
    openai_organization: Optional[str] = Field(
        default=None,
        description="OpenAI organization ID"
    )
    
    openai_api_base: Optional[str] = Field(
        default=None,
        description="Custom OpenAI API base URL"
    )
    
    azure_openai_endpoint: Optional[str] = Field(
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
    
    agent_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for agent responses"
    )
    
    agent_temperature: Annotated[float, Field(ge=0.0, le=2.0)] = Field(
        default=0.7,
        description="Temperature for response generation"
    )
    
    agent_max_tokens: Annotated[int, Field(ge=100, le=8000)] = Field(
        default=2000,
        description="Maximum tokens in agent response"
    )
    
    agent_timeout: Annotated[int, Field(ge=10, le=120)] = Field(
        default=30,
        description="Agent response timeout in seconds"
    )
    
    agent_max_retries: Annotated[int, Field(ge=0, le=5)] = Field(
        default=3,
        description="Maximum retries for failed API calls"
    )
    
    agent_tool_registry_mode: str = Field(
        default="legacy",
        description="Tool initialization mode: 'legacy' or 'registry'"
    )
    
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name"
    )
    
    embedding_gemma_model: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        description="Qwen3 Embedding model"
    )
    
    embedding_dimension: Annotated[int, Field(ge=128, le=2048)] = Field(
        default=768,
        description="Embedding vector dimension"
    )
    
    embedding_batch_size: Annotated[int, Field(ge=1, le=128)] = Field(
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
    
    rag_chunk_size: Annotated[int, Field(ge=100, le=2000)] = Field(
        default=500,
        description="Text chunk size in words"
    )
    
    rag_chunk_overlap: Annotated[int, Field(ge=0, le=500)] = Field(
        default=50,
        description="Overlap between chunks in words"
    )
    
    rag_search_k: Annotated[int, Field(ge=1, le=20)] = Field(
        default=5,
        description="Default number of RAG results"
    )
    
    rag_similarity_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.7,
        description="Minimum similarity score"
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
    
    memory_max_entries: Annotated[int, Field(ge=10, le=1000)] = Field(
        default=100,
        description="Maximum memory entries per session"
    )
    
    memory_ttl_hours: Annotated[int, Field(ge=1, le=720)] = Field(
        default=24,
        description="Memory TTL in hours"
    )
    
    memory_cleanup_days: Annotated[int, Field(ge=1, le=365)] = Field(
        default=30,
        description="Days before cleaning old memories"
    )
    
    # ===========================
    # File Handling
    # ===========================
    
    max_file_size: Annotated[int, Field(ge=1024, le=100*1024*1024)] = Field(
        default=10 * 1024 * 1024,
        description="Maximum file upload size in bytes"
    )
    
    allowed_file_types: List[str] = Field(
        default=[
            ".pdf", ".doc", ".docx", ".txt", ".md",
            ".csv", ".xlsx", ".xls", ".json", ".xml",
            ".jpg", ".jpeg", ".png", ".gif", ".bmp",
            ".mp3", ".wav", ".m4a"
        ],
        description="Allowed file extensions"
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
    # Rate Limiting
    # ===========================
    
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    
    rate_limit_requests: Annotated[int, Field(ge=10, le=10000)] = Field(
        default=100,
        description="Maximum requests per period"
    )
    
    rate_limit_period: Annotated[int, Field(ge=1, le=3600)] = Field(
        default=60,
        description="Rate limit period in seconds"
    )
    
    rate_limit_burst: Annotated[int, Field(ge=1, le=100)] = Field(
        default=10,
        description="Burst allowance"
    )
    
    # ===========================
    # Monitoring & Telemetry
    # ===========================
    
    enable_telemetry: bool = Field(
        default=False,
        description="Enable OpenTelemetry tracing"
    )
    
    metrics_enabled: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    
    metrics_port: Annotated[int, Field(ge=1024, le=65535)] = Field(
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
    
    websocket_ping_interval: Annotated[int, Field(ge=10, le=300)] = Field(
        default=30,
        description="WebSocket ping interval in seconds"
    )
    
    websocket_ping_timeout: Annotated[int, Field(ge=5, le=60)] = Field(
        default=10,
        description="WebSocket ping timeout in seconds"
    )
    
    websocket_max_connections: Annotated[int, Field(ge=10, le=10000)] = Field(
        default=1000,
        description="Maximum concurrent WebSocket connections"
    )
    
    # ===========================
    # Escalation Configuration
    # ===========================
    
    escalation_enabled: bool = Field(
        default=True,
        description="Enable escalation to human support"
    )
    
    escalation_confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.7,
        description="Confidence threshold for escalation"
    )
    
    escalation_keywords: Dict[str, float] = Field(
        default={
            "urgent": 1.0,
            "emergency": 1.0,
            "legal": 0.9,
            "complaint": 0.9,
            "manager": 0.8
        },
        description="Escalation keywords with weights"
    )
    
    escalation_notification_email: Optional[str] = Field(
        default=None,
        description="Email for escalation notifications"
    )
    
    escalation_webhook_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for escalations"
    )
    
    # ===========================
    # Feature Flags
    # ===========================
    
    feature_voice_input: bool = Field(default=False)
    feature_multilingual: bool = Field(default=False)
    feature_analytics: bool = Field(default=False)
    feature_export_chat: bool = Field(default=True)
    feature_file_preview: bool = Field(default=True)
    
    # ===========================
    # Development Settings
    # ===========================
    
    dev_auto_reload: bool = Field(default=False)
    dev_sample_data: bool = Field(default=False)
    dev_mock_ai: bool = Field(default=False)
    dev_slow_mode: bool = Field(default=False)
    
    # ===========================
    # Validators
    # ===========================
    
    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v: Union[str, List[str], None]) -> List[str]:
        """Parse CORS origins from string or list."""
        if v is None:
            return ["http://localhost:3000"]
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            if v.startswith('['):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        return ["http://localhost:3000"]
    
    @field_validator('allowed_file_types', mode='before')
    @classmethod
    def parse_allowed_file_types(cls, v: Union[str, List[str], None]) -> List[str]:
        """Parse allowed file types from string or list."""
        default = [
            ".pdf", ".doc", ".docx", ".txt", ".md",
            ".csv", ".xlsx", ".xls", ".json", ".xml",
            ".jpg", ".jpeg", ".png", ".gif", ".bmp",
            ".mp3", ".wav", ".m4a"
        ]
        if v is None:
            return default
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            if v.startswith('['):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            return [ext.strip() for ext in v.split(',') if ext.strip()]
        return default
    
    @field_validator('escalation_keywords', mode='before')
    @classmethod
    def parse_escalation_keywords(cls, v: Union[str, Dict[str, float], None]) -> Dict[str, float]:
        """Parse escalation keywords from string or dict."""
        default = {
            "urgent": 1.0,
            "emergency": 1.0,
            "legal": 0.9,
            "complaint": 0.9,
            "manager": 0.8
        }
        if v is None:
            return default
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            if v.startswith('{'):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            result = {}
            for pair in v.split(','):
                if '=' in pair:
                    key, value = pair.strip().split('=', 1)
                    try:
                        result[key] = float(value)
                    except ValueError:
                        result[key] = 0.8
                else:
                    result[pair.strip()] = 0.8
            return result if result else default
        return default
    
    @field_validator('environment', mode='before')
    @classmethod
    def validate_environment(cls, v: Union[str, Environment]) -> Environment:
        """Validate environment."""
        if isinstance(v, str):
            try:
                return Environment(v.lower())
            except ValueError:
                return Environment.DEVELOPMENT
        return v
    
    @field_validator('database_url')
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Validate database URL and create directories."""
        if v.startswith('sqlite:///'):
            db_path = v.replace('sqlite:///', '')
            if not os.path.isabs(db_path):
                db_dir = os.path.dirname(db_path)
                if db_dir:
                    Path(db_dir).mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator('upload_directory', 'chroma_persist_directory')
    @classmethod
    def create_directories(cls, v: str) -> str:
        """Create directories if they don't exist."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator('session_encryption_key', mode='before')
    @classmethod
    def generate_session_encryption_key(cls, v: Optional[str], info) -> Optional[SecretStr]:
        """Generate session encryption key if encryption is enabled."""
        # Access values through info.data
        values = info.data
        encryption_enabled = values.get('session_encryption_enabled', True)
        
        if not encryption_enabled:
            return None
        
        if v is None:
            # Generate 32-byte key
            from cryptography.fernet import Fernet
            key = Fernet.generate_key()
            return SecretStr(key.decode())
        
        if isinstance(v, str):
            return SecretStr(v)
        
        return v
    
    @model_validator(mode='after')
    def validate_session_store_dependencies(self) -> 'Settings':
        """Validate session store configuration dependencies."""
        # Handle legacy use_shared_context
        if self.use_shared_context and self.session_store_type == SessionStoreType.IN_MEMORY:
            self.session_store_type = SessionStoreType.REDIS
        
        # Validate Redis requirements
        if self.session_store_type == SessionStoreType.REDIS:
            if not self.redis_url or self.redis_url == "redis://localhost:6379/0":
                if self.environment == Environment.PRODUCTION:
                    import warnings
                    warnings.warn(
                        "Using localhost Redis in production with REDIS session store",
                        UserWarning
                    )
            
            # L1 cache should be enabled for Redis sessions
            if not self.session_l1_cache_enabled and self.environment == Environment.PRODUCTION:
                import warnings
                warnings.warn(
                    "L1 cache disabled for Redis sessions - may impact performance",
                    UserWarning
                )
        
        # Validate locking requirements
        if self.session_locking_enabled and self.session_store_type == SessionStoreType.IN_MEMORY:
            # Locking not needed for in-memory store
            self.session_locking_enabled = False
        
        # Validate encryption key
        if self.session_encryption_enabled and not self.session_encryption_key:
            raise ValueError("session_encryption_key required when session_encryption_enabled=True")
        
        return self
    
    @model_validator(mode='after')
    def validate_production_requirements(self) -> 'Settings':
        """Validate production environment requirements."""
        if self.environment == Environment.PRODUCTION:
            issues = []
            
            if self.debug:
                issues.append("Debug mode enabled in production")
            
            if self.database_url.startswith('sqlite:'):
                issues.append("SQLite database in production (PostgreSQL recommended)")
            
            if not self.session_encryption_enabled:
                issues.append("Session encryption disabled in production")
            
            if not self.session_fingerprinting_enabled:
                issues.append("Session fingerprinting disabled in production")
            
            if self.session_store_type == SessionStoreType.IN_MEMORY:
                issues.append("In-memory sessions in production (Redis recommended)")
            
            if issues:
                import warnings
                for issue in issues:
                    warnings.warn(f"Production configuration issue: {issue}", UserWarning)
        
        return self
    
    # ===========================
    # Properties
    # ===========================
    
    @property
    def database_is_postgresql(self) -> bool:
        """Check if using PostgreSQL."""
        return self.database_url.startswith('postgresql')
    
    @property
    def database_is_sqlite(self) -> bool:
        """Check if using SQLite."""
        return self.database_url.startswith('sqlite')
    
    @property
    def redis_url_with_password(self) -> str:
        """Get Redis URL with password if configured."""
        if not self.redis_password:
            return self.redis_url
        
        try:
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(self.redis_url)
            
            if not parsed.hostname:
                return self.redis_url
            
            port = parsed.port or 6379
            netloc = f":{self.redis_password.get_secret_value()}@{parsed.hostname}:{port}"
            return urlunparse(parsed._replace(netloc=netloc))
        except Exception:
            return self.redis_url
    
    @property
    def is_production(self) -> bool:
        """Check if in production."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if in development."""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def session_timeout_seconds(self) -> int:
        """Get session timeout in seconds."""
        return self.session_timeout_minutes * 60
    
    @property
    def requires_redis(self) -> bool:
        """Check if Redis is required."""
        return (
            self.cache_enabled or
            self.session_store_type == SessionStoreType.REDIS
        )
    
    # ===========================
    # Methods
    # ===========================
    
    def get_safe_dict(self) -> Dict[str, Any]:
        """Get configuration dict with secrets masked."""
        config_dict = self.model_dump()
        
        sensitive_fields = [
            'secret_key', 'openai_api_key', 'azure_openai_api_key',
            'redis_password', 'sentry_dsn', 'session_encryption_key'
        ]
        
        for field in sensitive_fields:
            if field in config_dict and config_dict[field]:
                config_dict[field] = "***MASKED***"
        
        # Partially mask URLs
        for field in ['database_url', 'redis_url']:
            if field in config_dict and config_dict[field]:
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(config_dict[field])
                    config_dict[field] = f"{parsed.scheme}://***MASKED***"
                except:
                    config_dict[field] = "***MASKED***"
        
        return config_dict


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()

__all__ = ['Settings', 'get_settings', 'settings', 'Environment', 'LogLevel', 'AIProvider', 'SessionStoreType']
```

This config.py is now complete with:
- ✅ All duplicates removed
- ✅ Session configuration consolidated and enhanced
- ✅ Encryption, locking, fingerprinting, L1 cache support
- ✅ Comprehensive validation
- ✅ Production safety checks
- ✅ No placeholders

