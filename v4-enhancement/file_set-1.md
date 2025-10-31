# backend/app/utils/retry.py
```py
"""
Retry and resilience utilities for database and external service calls.
Implements retry logic, exponential backoff, and circuit breaker pattern.

Version: 1.0.0
"""
import asyncio
import logging
import time
import functools
from typing import Callable, Optional, Tuple, Type, Union, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RetryStrategy(str, Enum):
    """Retry strategy enumeration."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retry_on_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    retry_on_result: Optional[Callable[[Any], bool]] = None


class CircuitBreakerState(str, Enum):
    """Circuit breaker state."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failures exceeded threshold, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker implementation to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are rejected
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            success_threshold: Successes needed to close circuit from half-open
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_recovery():
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute async function through circuit breaker.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_recovery():
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.success_count = 0
                logger.info("Circuit breaker CLOSED after successful recovery")
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_count = 0
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(
                f"Circuit breaker OPENED after {self.failure_count} failures"
            )
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info("Circuit breaker manually reset to CLOSED")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


def calculate_retry_delay(
    attempt: int,
    config: RetryConfig
) -> float:
    """
    Calculate delay before next retry attempt.
    
    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration
        
    Returns:
        Delay in seconds
    """
    if config.strategy == RetryStrategy.FIXED:
        return config.initial_delay
    
    elif config.strategy == RetryStrategy.LINEAR:
        delay = config.initial_delay * (attempt + 1)
    
    elif config.strategy == RetryStrategy.EXPONENTIAL:
        delay = config.initial_delay * (config.exponential_base ** attempt)
    
    else:
        delay = config.initial_delay
    
    # Cap at max delay
    return min(delay, config.max_delay)


def retry(config: Optional[RetryConfig] = None):
    """
    Decorator for retrying function calls with configurable backoff.
    
    Args:
        config: Retry configuration
        
    Example:
        @retry(RetryConfig(max_attempts=3, strategy=RetryStrategy.EXPONENTIAL))
        def unreliable_function():
            ...
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    result = func(*args, **kwargs)
                    
                    # Check if result should trigger retry
                    if config.retry_on_result and config.retry_on_result(result):
                        if attempt < config.max_attempts - 1:
                            delay = calculate_retry_delay(attempt, config)
                            logger.warning(
                                f"{func.__name__} returned retry-worthy result, "
                                f"retrying in {delay:.2f}s (attempt {attempt + 1}/{config.max_attempts})"
                            )
                            time.sleep(delay)
                            continue
                    
                    return result
                    
                except config.retry_on_exceptions as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts - 1:
                        delay = calculate_retry_delay(attempt, config)
                        logger.warning(
                            f"{func.__name__} failed with {type(e).__name__}, "
                            f"retrying in {delay:.2f}s (attempt {attempt + 1}/{config.max_attempts}): {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {config.max_attempts} attempts: {e}"
                        )
            
            # All retries exhausted
            if last_exception:
                raise last_exception
            
            # This shouldn't happen, but just in case
            raise RuntimeError(f"{func.__name__} failed without exception")
        
        return wrapper
    return decorator


def async_retry(config: Optional[RetryConfig] = None):
    """
    Decorator for retrying async function calls with configurable backoff.
    
    Args:
        config: Retry configuration
        
    Example:
        @async_retry(RetryConfig(max_attempts=3))
        async def unreliable_async_function():
            ...
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    result = await func(*args, **kwargs)
                    
                    # Check if result should trigger retry
                    if config.retry_on_result and config.retry_on_result(result):
                        if attempt < config.max_attempts - 1:
                            delay = calculate_retry_delay(attempt, config)
                            logger.warning(
                                f"{func.__name__} returned retry-worthy result, "
                                f"retrying in {delay:.2f}s (attempt {attempt + 1}/{config.max_attempts})"
                            )
                            await asyncio.sleep(delay)
                            continue
                    
                    return result
                    
                except config.retry_on_exceptions as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts - 1:
                        delay = calculate_retry_delay(attempt, config)
                        logger.warning(
                            f"{func.__name__} failed with {type(e).__name__}, "
                            f"retrying in {delay:.2f}s (attempt {attempt + 1}/{config.max_attempts}): {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {config.max_attempts} attempts: {e}"
                        )
            
            # All retries exhausted
            if last_exception:
                raise last_exception
            
            raise RuntimeError(f"{func.__name__} failed without exception")
        
        return wrapper
    return decorator


__all__ = [
    'RetryConfig',
    'RetryStrategy',
    'CircuitBreaker',
    'CircuitBreakerState',
    'CircuitBreakerOpenError',
    'retry',
    'async_retry',
    'calculate_retry_delay'
]

```

# backend/app/utils/__init__.py
```py
"""
Utility modules for the application.
Provides encryption, retry logic, telemetry, and middleware.

Version: 1.0.0
"""

# Encryption utilities
try:
    from .encryption import (
        SessionEncryption,
        TimestampedEncryption,
        EncryptionError,
        create_encryption_instance
    )
    ENCRYPTION_AVAILABLE = True
except ImportError:
    SessionEncryption = None
    TimestampedEncryption = None
    EncryptionError = None
    create_encryption_instance = None
    ENCRYPTION_AVAILABLE = False

# Retry utilities
try:
    from .retry import (
        RetryConfig,
        RetryStrategy,
        CircuitBreaker,
        CircuitBreakerState,
        CircuitBreakerOpenError,
        retry,
        async_retry,
        calculate_retry_delay
    )
    RETRY_AVAILABLE = True
except ImportError:
    RetryConfig = None
    RetryStrategy = None
    CircuitBreaker = None
    CircuitBreakerState = None
    CircuitBreakerOpenError = None
    retry = None
    async_retry = None
    calculate_retry_delay = None
    RETRY_AVAILABLE = False

# Telemetry utilities
try:
    from .telemetry import (
        setup_telemetry,
        metrics_collector,
        get_tracer,
        trace_async,
        trace_sync
    )
    TELEMETRY_AVAILABLE = True
except ImportError:
    setup_telemetry = None
    metrics_collector = None
    get_tracer = None
    trace_async = None
    trace_sync = None
    TELEMETRY_AVAILABLE = False

# Middleware
try:
    from .middleware import (
        RequestIDMiddleware,
        TimingMiddleware,
        RateLimitMiddleware,
        ErrorHandlingMiddleware
    )
    MIDDLEWARE_AVAILABLE = True
except ImportError:
    RequestIDMiddleware = None
    TimingMiddleware = None
    RateLimitMiddleware = None
    ErrorHandlingMiddleware = None
    MIDDLEWARE_AVAILABLE = False


__all__ = [
    # Encryption
    'SessionEncryption',
    'TimestampedEncryption',
    'EncryptionError',
    'create_encryption_instance',
    
    # Retry
    'RetryConfig',
    'RetryStrategy',
    'CircuitBreaker',
    'CircuitBreakerState',
    'CircuitBreakerOpenError',
    'retry',
    'async_retry',
    'calculate_retry_delay',
    
    # Telemetry
    'setup_telemetry',
    'metrics_collector',
    'get_tracer',
    'trace_async',
    'trace_sync',
    
    # Middleware
    'RequestIDMiddleware',
    'TimingMiddleware',
    'RateLimitMiddleware',
    'ErrorHandlingMiddleware',
    
    # Availability flags
    'ENCRYPTION_AVAILABLE',
    'RETRY_AVAILABLE',
    'TELEMETRY_AVAILABLE',
    'MIDDLEWARE_AVAILABLE'
]

```

# backend/app/utils/encryption.py
```py
"""
Session data encryption utilities.
Provides secure encryption/decryption for sensitive session data.

Version: 1.0.0
"""
import logging
import base64
from typing import Optional, Union
from datetime import datetime, timedelta

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class EncryptionError(Exception):
    """Raised when encryption/decryption fails."""
    pass


class SessionEncryption:
    """
    Session data encryption using Fernet (symmetric encryption).
    
    Features:
    - AES-128 encryption in CBC mode
    - HMAC for integrity verification
    - Automatic key rotation support
    - Secure key derivation from passwords
    """
    
    def __init__(self, encryption_key: Optional[Union[str, bytes]] = None):
        """
        Initialize encryption with key.
        
        Args:
            encryption_key: Base64-encoded Fernet key or password
        """
        self.cipher: Optional[Fernet] = None
        self.key: Optional[bytes] = None
        
        if encryption_key:
            self._initialize_cipher(encryption_key)
    
    def _initialize_cipher(self, encryption_key: Union[str, bytes]) -> None:
        """
        Initialize Fernet cipher with key.
        
        Args:
            encryption_key: Encryption key (base64 or password)
        """
        try:
            # Convert string to bytes
            if isinstance(encryption_key, str):
                key_bytes = encryption_key.encode()
            else:
                key_bytes = encryption_key
            
            # Try to use as Fernet key directly
            try:
                self.cipher = Fernet(key_bytes)
                self.key = key_bytes
                logger.debug("Encryption cipher initialized with provided key")
            except Exception:
                # If not valid Fernet key, derive from password
                logger.debug("Deriving encryption key from password")
                self.key = self._derive_key_from_password(key_bytes)
                self.cipher = Fernet(self.key)
                
        except Exception as e:
            logger.error(f"Failed to initialize encryption cipher: {e}")
            raise EncryptionError(f"Cipher initialization failed: {e}")
    
    def _derive_key_from_password(
        self,
        password: bytes,
        salt: Optional[bytes] = None
    ) -> bytes:
        """
        Derive Fernet key from password using PBKDF2.
        
        Args:
            password: Password to derive key from
            salt: Salt for key derivation (generated if None)
            
        Returns:
            Base64-encoded Fernet key
        """
        if salt is None:
            # Use fixed salt for deterministic key derivation
            # In production, consider storing salt separately
            salt = b"customer_support_ai_salt_v1"
        
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    @staticmethod
    def generate_key() -> str:
        """
        Generate a new Fernet encryption key.
        
        Returns:
            Base64-encoded key as string
        """
        key = Fernet.generate_key()
        return key.decode()
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
            
        Raises:
            EncryptionError: If encryption fails
        """
        if not self.cipher:
            raise EncryptionError("Encryption cipher not initialized")
        
        try:
            # Convert string to bytes
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Encrypt
            encrypted = self.cipher.encrypt(data)
            
            logger.debug(f"Encrypted {len(data)} bytes -> {len(encrypted)} bytes")
            return encrypted
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise EncryptionError(f"Encryption failed: {e}")
    
    def decrypt(self, encrypted_data: Union[str, bytes]) -> bytes:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data
            
        Raises:
            EncryptionError: If decryption fails
        """
        if not self.cipher:
            raise EncryptionError("Encryption cipher not initialized")
        
        try:
            # Convert string to bytes if needed
            if isinstance(encrypted_data, str):
                encrypted_data = encrypted_data.encode('utf-8')
            
            # Decrypt
            decrypted = self.cipher.decrypt(encrypted_data)
            
            logger.debug(f"Decrypted {len(encrypted_data)} bytes -> {len(decrypted)} bytes")
            return decrypted
            
        except InvalidToken:
            logger.error("Decryption failed: Invalid token (wrong key or corrupted data)")
            raise EncryptionError("Decryption failed: Invalid token")
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise EncryptionError(f"Decryption failed: {e}")
    
    def encrypt_string(self, data: str) -> str:
        """
        Encrypt string and return base64-encoded result.
        
        Args:
            data: String to encrypt
            
        Returns:
            Base64-encoded encrypted string
        """
        encrypted = self.encrypt(data)
        return base64.b64encode(encrypted).decode('utf-8')
    
    def decrypt_string(self, encrypted_data: str) -> str:
        """
        Decrypt base64-encoded encrypted string.
        
        Args:
            encrypted_data: Base64-encoded encrypted string
            
        Returns:
            Decrypted string
        """
        encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
        decrypted = self.decrypt(encrypted_bytes)
        return decrypted.decode('utf-8')
    
    def rotate_key(self, new_key: Union[str, bytes]) -> None:
        """
        Rotate encryption key.
        
        Args:
            new_key: New encryption key
            
        Note:
            Existing encrypted data will need to be re-encrypted with new key
        """
        logger.info("Rotating encryption key")
        self._initialize_cipher(new_key)
    
    def verify_key(self, test_data: str = "test") -> bool:
        """
        Verify encryption key by performing encrypt/decrypt roundtrip.
        
        Args:
            test_data: Test data to use
            
        Returns:
            True if key is valid
        """
        try:
            encrypted = self.encrypt(test_data)
            decrypted = self.decrypt(encrypted)
            return decrypted.decode('utf-8') == test_data
        except Exception as e:
            logger.error(f"Key verification failed: {e}")
            return False


class TimestampedEncryption(SessionEncryption):
    """
    Encryption with built-in timestamp validation.
    Prevents replay attacks by validating encryption age.
    """
    
    def __init__(
        self,
        encryption_key: Optional[Union[str, bytes]] = None,
        max_age_seconds: int = 3600
    ):
        """
        Initialize timestamped encryption.
        
        Args:
            encryption_key: Encryption key
            max_age_seconds: Maximum age for encrypted data
        """
        super().__init__(encryption_key)
        self.max_age_seconds = max_age_seconds
    
    def encrypt_with_timestamp(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data with timestamp.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data with embedded timestamp
        """
        if not self.cipher:
            raise EncryptionError("Encryption cipher not initialized")
        
        try:
            # Convert to bytes
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Encrypt with TTL
            encrypted = self.cipher.encrypt_at_time(
                data,
                current_time=int(datetime.utcnow().timestamp())
            )
            
            return encrypted
            
        except Exception as e:
            logger.error(f"Timestamped encryption failed: {e}")
            raise EncryptionError(f"Timestamped encryption failed: {e}")
    
    def decrypt_with_timestamp(self, encrypted_data: Union[str, bytes]) -> bytes:
        """
        Decrypt data and validate timestamp.
        
        Args:
            encrypted_data: Encrypted data with timestamp
            
        Returns:
            Decrypted data
            
        Raises:
            EncryptionError: If data is too old or decryption fails
        """
        if not self.cipher:
            raise EncryptionError("Encryption cipher not initialized")
        
        try:
            # Convert to bytes
            if isinstance(encrypted_data, str):
                encrypted_data = encrypted_data.encode('utf-8')
            
            # Decrypt with TTL validation
            decrypted = self.cipher.decrypt_at_time(
                encrypted_data,
                ttl=self.max_age_seconds,
                current_time=int(datetime.utcnow().timestamp())
            )
            
            return decrypted
            
        except InvalidToken as e:
            if "too old" in str(e).lower():
                logger.warning(f"Encrypted data expired (max_age={self.max_age_seconds}s)")
                raise EncryptionError("Encrypted data has expired")
            else:
                logger.error(f"Invalid token: {e}")
                raise EncryptionError("Invalid encrypted data")
        except Exception as e:
            logger.error(f"Timestamped decryption failed: {e}")
            raise EncryptionError(f"Timestamped decryption failed: {e}")


def create_encryption_instance(
    encryption_key: Optional[Union[str, bytes]] = None,
    use_timestamp: bool = False,
    max_age_seconds: int = 3600
) -> Union[SessionEncryption, TimestampedEncryption]:
    """
    Factory function to create encryption instance.
    
    Args:
        encryption_key: Encryption key
        use_timestamp: Whether to use timestamped encryption
        max_age_seconds: Maximum age for timestamped encryption
        
    Returns:
        Encryption instance
    """
    if use_timestamp:
        return TimestampedEncryption(encryption_key, max_age_seconds)
    else:
        return SessionEncryption(encryption_key)


__all__ = [
    'SessionEncryption',
    'TimestampedEncryption',
    'EncryptionError',
    'create_encryption_instance'
]

```

# backend/app/config.py
```py
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

