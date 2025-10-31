# 🚀 Complete File Implementations (Continued - Phase 2)

## File 5: `backend/app/session/validators.py` (NEW)

**Checklist:**
- [x] Implement Pydantic-based session data validation
- [x] Add comprehensive field validators
- [x] Implement metadata validation
- [x] Add security validation (fingerprinting)
- [x] Include serialization validation

```python
"""
Session data validation using Pydantic.
Ensures session data integrity and type safety.

Version: 1.0.0
"""
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class SessionData(BaseModel):
    """
    Validated session data structure.
    
    Features:
    - Type validation
    - Constraint validation
    - JSON serialization validation
    - Fingerprint security validation
    """
    
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique session identifier"
    )
    
    user_id: Optional[str] = Field(
        None,
        max_length=255,
        description="User identifier"
    )
    
    thread_id: Optional[str] = Field(
        None,
        max_length=255,
        description="Thread identifier for conversation context"
    )
    
    message_count: int = Field(
        default=0,
        ge=0,
        le=100000,
        description="Number of messages in session"
    )
    
    escalated: bool = Field(
        default=False,
        description="Whether session has been escalated"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional session metadata"
    )
    
    request_id: Optional[str] = Field(
        None,
        max_length=255,
        description="Current request correlation ID"
    )
    
    created_at: Optional[datetime] = Field(
        None,
        description="Session creation timestamp"
    )
    
    updated_at: Optional[datetime] = Field(
        None,
        description="Last update timestamp"
    )
    
    last_activity: Optional[datetime] = Field(
        None,
        description="Last activity timestamp"
    )
    
    fingerprint: Optional[str] = Field(
        None,
        max_length=64,
        description="Session fingerprint for security"
    )
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        use_enum_values = True
        validate_assignment = True
    
    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure metadata is JSON-serializable.
        
        Args:
            v: Metadata dictionary
            
        Returns:
            Validated metadata
            
        Raises:
            ValueError: If metadata is not JSON-serializable
        """
        try:
            # Test JSON serialization
            json_str = json.dumps(v)
            
            # Check size
            if len(json_str) > 1_000_000:  # 1MB limit
                raise ValueError("Metadata exceeds maximum size (1MB)")
            
            return v
            
        except (TypeError, ValueError) as e:
            raise ValueError(f"Metadata must be JSON-serializable: {e}")
    
    @field_validator('message_count')
    @classmethod
    def validate_message_count(cls, v: int) -> int:
        """
        Validate message count is reasonable.
        
        Args:
            v: Message count
            
        Returns:
            Validated message count
        """
        if v < 0:
            raise ValueError("Message count cannot be negative")
        
        if v > 100000:
            raise ValueError("Message count exceeds maximum (100,000)")
        
        return v
    
    @field_validator('session_id', 'user_id', 'thread_id', 'request_id')
    @classmethod
    def validate_identifiers(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate identifier fields.
        
        Args:
            v: Identifier value
            
        Returns:
            Validated identifier
        """
        if v is None:
            return v
        
        # Check for SQL injection attempts
        dangerous_chars = ["'", '"', ';', '--', '/*', '*/', 'DROP', 'DELETE']
        v_upper = v.upper()
        
        for char in dangerous_chars:
            if char in v_upper:
                logger.warning(f"Potentially dangerous characters in identifier: {v}")
                raise ValueError("Invalid characters in identifier")
        
        # Check length
        if len(v) > 255:
            raise ValueError("Identifier too long (max 255 characters)")
        
        return v
    
    @field_validator('fingerprint')
    @classmethod
    def validate_fingerprint(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate session fingerprint format.
        
        Args:
            v: Fingerprint hash
            
        Returns:
            Validated fingerprint
        """
        if v is None:
            return v
        
        # Should be a hex string (SHA-256 = 64 chars)
        if not all(c in '0123456789abcdef' for c in v.lower()):
            raise ValueError("Fingerprint must be hexadecimal")
        
        if len(v) not in [32, 40, 64]:  # MD5, SHA-1, SHA-256
            raise ValueError("Invalid fingerprint length")
        
        return v.lower()
    
    @model_validator(mode='after')
    def validate_timestamps(self) -> 'SessionData':
        """
        Validate timestamp consistency.
        
        Returns:
            Validated session data
        """
        # Ensure created_at <= updated_at <= last_activity
        if self.created_at and self.updated_at:
            if self.updated_at < self.created_at:
                raise ValueError("updated_at cannot be before created_at")
        
        if self.updated_at and self.last_activity:
            if self.last_activity < self.updated_at:
                # This is acceptable - last_activity might be updated separately
                pass
        
        # Auto-set timestamps if missing
        now = datetime.utcnow()
        
        if not self.created_at:
            self.created_at = now
        
        if not self.updated_at:
            self.updated_at = now
        
        if not self.last_activity:
            self.last_activity = now
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        data = self.model_dump()
        
        # Convert datetime objects to ISO format strings
        for key in ['created_at', 'updated_at', 'last_activity']:
            if data.get(key) and isinstance(data[key], datetime):
                data[key] = data[key].isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionData':
        """
        Create SessionData from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            SessionData instance
        """
        # Make a copy to avoid modifying input
        data = dict(data)
        
        # Convert ISO format strings to datetime
        for key in ['created_at', 'updated_at', 'last_activity']:
            if key in data and data[key] is not None:
                if isinstance(data[key], str):
                    try:
                        data[key] = datetime.fromisoformat(data[key])
                    except ValueError as e:
                        logger.warning(f"Invalid datetime string for {key}: {data[key]}")
                        data[key] = None
                elif not isinstance(data[key], datetime):
                    logger.warning(f"Unexpected type for {key}: {type(data[key])}")
                    data[key] = None
        
        return cls(**data)
    
    def to_json(self) -> str:
        """
        Serialize to JSON string.
        
        Returns:
            JSON string
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SessionData':
        """
        Deserialize from JSON string.
        
        Args:
            json_str: JSON string
            
        Returns:
            SessionData instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def update_activity(self) -> None:
        """Update last_activity and updated_at timestamps."""
        now = datetime.utcnow()
        self.last_activity = now
        self.updated_at = now
    
    def increment_message_count(self, delta: int = 1) -> int:
        """
        Increment message count.
        
        Args:
            delta: Amount to increment by
            
        Returns:
            New message count
        """
        self.message_count += delta
        self.update_activity()
        return self.message_count
    
    def create_fingerprint(self, ip_address: str, user_agent: str) -> str:
        """
        Create session fingerprint from IP and user agent.
        
        Args:
            ip_address: Client IP address
            user_agent: Client user agent string
            
        Returns:
            Fingerprint hash
        """
        import hashlib
        
        data = f"{ip_address}:{user_agent}"
        fingerprint = hashlib.sha256(data.encode()).hexdigest()
        self.fingerprint = fingerprint
        
        return fingerprint
    
    def verify_fingerprint(self, ip_address: str, user_agent: str) -> bool:
        """
        Verify session fingerprint matches current request.
        
        Args:
            ip_address: Client IP address
            user_agent: Client user agent string
            
        Returns:
            True if fingerprint matches
        """
        if not self.fingerprint:
            # No fingerprint set (backward compatibility)
            return True
        
        import hashlib
        
        data = f"{ip_address}:{user_agent}"
        current_fingerprint = hashlib.sha256(data.encode()).hexdigest()
        
        return self.fingerprint == current_fingerprint


class SessionMetrics(BaseModel):
    """Session metrics for monitoring."""
    
    total_messages: int = Field(ge=0)
    avg_response_time: float = Field(ge=0.0)
    escalation_rate: float = Field(ge=0.0, le=1.0)
    active_sessions: int = Field(ge=0)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class SessionFilter(BaseModel):
    """Filter criteria for session queries."""
    
    user_id: Optional[str] = None
    escalated: Optional[bool] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    min_messages: Optional[int] = Field(None, ge=0)
    max_messages: Optional[int] = Field(None, ge=0)
    
    @model_validator(mode='after')
    def validate_date_range(self) -> 'SessionFilter':
        """Validate date range."""
        if self.created_after and self.created_before:
            if self.created_before < self.created_after:
                raise ValueError("created_before must be after created_after")
        
        return self
    
    @model_validator(mode='after')
    def validate_message_range(self) -> 'SessionFilter':
        """Validate message count range."""
        if self.min_messages is not None and self.max_messages is not None:
            if self.max_messages < self.min_messages:
                raise ValueError("max_messages must be >= min_messages")
        
        return self


__all__ = ['SessionData', 'SessionMetrics', 'SessionFilter']
```

---

## File 6: `backend/app/session/session_store.py` (Enhanced)

**Checklist:**
- [x] Update to use Pydantic SessionData
- [x] Add batch operations
- [x] Improve abstract interface
- [x] Add comprehensive docstrings
- [x] Include health monitoring methods

```python
"""
Abstract session store interface.
Defines the contract for session persistence implementations.

Version: 2.0.0 (Enhanced with validation and batch operations)
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime

from .validators import SessionData


class SessionStore(ABC):
    """
    Abstract base class for session storage.
    
    Implementations must provide thread-safe/async-safe operations for:
    - Getting session data
    - Setting session data
    - Updating session data (atomic operations for counters)
    - Deleting session data
    - Listing active sessions
    - Batch operations
    """
    
    @abstractmethod
    async def get(self, session_id: str) -> Optional[SessionData]:
        """
        Get session data by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionData or None if not found
        """
        pass
    
    @abstractmethod
    async def set(
        self,
        session_id: str,
        session_data: SessionData,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set session data.
        
        Args:
            session_id: Session identifier
            session_data: Session data to store
            ttl: Time-to-live in seconds (optional)
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def update(
        self,
        session_id: str,
        updates: Dict[str, Any],
        atomic: bool = False
    ) -> bool:
        """
        Update session data.
        
        Args:
            session_id: Session identifier
            updates: Fields to update
            atomic: Whether to use atomic operations
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """
        Delete session data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        """
        Check if session exists.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session exists
        """
        pass
    
    @abstractmethod
    async def list_active(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[str]:
        """
        List active session IDs.
        
        Args:
            limit: Maximum number of sessions to return
            offset: Offset for pagination
            
        Returns:
            List of session IDs
        """
        pass
    
    @abstractmethod
    async def count_active(self) -> int:
        """
        Count active sessions.
        
        Returns:
            Number of active sessions
        """
        pass
    
    @abstractmethod
    async def cleanup_expired(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        pass
    
    @abstractmethod
    async def increment_counter(
        self,
        session_id: str,
        field: str,
        delta: int = 1
    ) -> int:
        """
        Atomically increment a counter field.
        
        Args:
            session_id: Session identifier
            field: Field name to increment
            delta: Increment value
            
        Returns:
            New counter value
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get session store statistics.
        
        Returns:
            Dictionary with statistics
        """
        pass
    
    async def get_batch(self, session_ids: List[str]) -> Dict[str, Optional[SessionData]]:
        """
        Get multiple sessions in batch.
        
        Args:
            session_ids: List of session identifiers
            
        Returns:
            Dictionary mapping session_id to SessionData
        """
        result = {}
        for session_id in session_ids:
            result[session_id] = await self.get(session_id)
        return result
    
    async def set_batch(
        self,
        sessions: Dict[str, SessionData],
        ttl: Optional[int] = None
    ) -> Dict[str, bool]:
        """
        Set multiple sessions in batch.
        
        Args:
            sessions: Dictionary mapping session_id to SessionData
            ttl: Time-to-live in seconds
            
        Returns:
            Dictionary mapping session_id to success status
        """
        result = {}
        for session_id, session_data in sessions.items():
            result[session_id] = await self.set(session_id, session_data, ttl)
        return result
    
    async def delete_batch(self, session_ids: List[str]) -> Dict[str, bool]:
        """
        Delete multiple sessions in batch.
        
        Args:
            session_ids: List of session identifiers
            
        Returns:
            Dictionary mapping session_id to deletion status
        """
        result = {}
        for session_id in session_ids:
            result[session_id] = await self.delete(session_id)
        return result
    
    async def touch(self, session_id: str) -> bool:
        """
        Update session's last_activity timestamp without other changes.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful
        """
        session_data = await self.get(session_id)
        if not session_data:
            return False
        
        session_data.update_activity()
        return await self.set(session_id, session_data)
    
    async def get_or_create(
        self,
        session_id: str,
        factory: Optional[callable] = None,
        ttl: Optional[int] = None
    ) -> SessionData:
        """
        Get session or create if not exists.
        
        Args:
            session_id: Session identifier
            factory: Optional factory function to create new session
            ttl: Time-to-live for new session
            
        Returns:
            SessionData instance
        """
        session_data = await self.get(session_id)
        
        if session_data:
            return session_data
        
        # Create new session
        if factory:
            session_data = factory(session_id)
        else:
            session_data = SessionData(
                session_id=session_id,
                created_at=datetime.utcnow()
            )
        
        await self.set(session_id, session_data, ttl)
        return session_data
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on session store.
        
        Returns:
            Dictionary with health status
        """
        try:
            # Try basic operations
            test_session_id = f"health_check_{datetime.utcnow().timestamp()}"
            test_data = SessionData(session_id=test_session_id)
            
            # Test set
            set_success = await self.set(test_session_id, test_data, ttl=10)
            
            # Test get
            retrieved = await self.get(test_session_id)
            get_success = retrieved is not None
            
            # Test delete
            delete_success = await self.delete(test_session_id)
            
            # Get stats
            stats = await self.get_stats()
            
            return {
                "healthy": set_success and get_success and delete_success,
                "operations": {
                    "set": set_success,
                    "get": get_success,
                    "delete": delete_success
                },
                "stats": stats
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }


__all__ = ['SessionStore', 'SessionData']
```

---

## File 7: `backend/app/session/distributed_lock.py` (NEW)

**Checklist:**
- [x] Implement Redis-based distributed lock
- [x] Add automatic lock release on timeout
- [x] Implement lock renewal
- [x] Add context manager support
- [x] Include deadlock prevention

```python
"""
Distributed locking implementation for session operations.
Uses Redis for distributed coordination across multiple instances.

Version: 1.0.0
"""
import asyncio
import logging
import uuid
from typing import Optional
from datetime import datetime, timedelta

try:
    import redis.asyncio as redis
    from redis.asyncio import Redis
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    Redis = None
    RedisError = Exception
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class LockAcquisitionError(Exception):
    """Raised when lock acquisition fails."""
    pass


class LockReleaseError(Exception):
    """Raised when lock release fails."""
    pass


class DistributedLock:
    """
    Distributed lock using Redis.
    
    Features:
    - Automatic expiration to prevent deadlocks
    - Unique lock identifiers to prevent accidental release
    - Lock renewal support for long operations
    - Context manager support
    """
    
    # Lua script for atomic lock release (only if we own it)
    RELEASE_SCRIPT = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
    else
        return 0
    end
    """
    
    # Lua script for atomic lock renewal
    RENEW_SCRIPT = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("expire", KEYS[1], ARGV[2])
    else
        return 0
    end
    """
    
    def __init__(
        self,
        redis_client: Redis,
        lock_name: str,
        timeout: int = 30,
        retry_attempts: int = 3,
        retry_delay: float = 0.1
    ):
        """
        Initialize distributed lock.
        
        Args:
            redis_client: Redis client instance
            lock_name: Name of the lock
            timeout: Lock timeout in seconds
            retry_attempts: Number of acquisition retry attempts
            retry_delay: Delay between retries in seconds
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available for distributed locking")
        
        self.redis_client = redis_client
        self.lock_name = f"lock:{lock_name}"
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Unique identifier for this lock instance
        self.lock_id: Optional[str] = None
        self.acquired: bool = False
        
        # Scripts
        self.release_script_sha: Optional[str] = None
        self.renew_script_sha: Optional[str] = None
    
    async def _load_scripts(self) -> None:
        """Load Lua scripts into Redis."""
        try:
            if not self.release_script_sha:
                self.release_script_sha = await self.redis_client.script_load(
                    self.RELEASE_SCRIPT
                )
            
            if not self.renew_script_sha:
                self.renew_script_sha = await self.redis_client.script_load(
                    self.RENEW_SCRIPT
                )
        except RedisError as e:
            logger.warning(f"Failed to load Lua scripts: {e}")
    
    async def acquire(self, blocking: bool = True) -> bool:
        """
        Acquire the lock.
        
        Args:
            blocking: If True, retry until acquired or max attempts reached
            
        Returns:
            True if lock acquired
            
        Raises:
            LockAcquisitionError: If lock cannot be acquired
        """
        if self.acquired:
            logger.warning(f"Lock {self.lock_name} already acquired")
            return True
        
        # Generate unique lock ID
        self.lock_id = str(uuid.uuid4())
        
        # Load scripts
        await self._load_scripts()
        
        attempts = self.retry_attempts if blocking else 1
        
        for attempt in range(attempts):
            try:
                # Try to acquire lock with SET NX (set if not exists)
                acquired = await self.redis_client.set(
                    self.lock_name,
                    self.lock_id,
                    nx=True,  # Only set if not exists
                    ex=self.timeout  # Expiration time
                )
                
                if acquired:
                    self.acquired = True
                    logger.debug(
                        f"Lock {self.lock_name} acquired "
                        f"(id={self.lock_id[:8]}, timeout={self.timeout}s)"
                    )
                    return True
                
                # Lock not acquired
                if blocking and attempt < attempts - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    logger.debug(
                        f"Lock {self.lock_name} busy, "
                        f"retrying in {delay:.2f}s (attempt {attempt + 1}/{attempts})"
                    )
                    await asyncio.sleep(delay)
                
            except RedisError as e:
                logger.error(f"Redis error acquiring lock {self.lock_name}: {e}")
                if attempt < attempts - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise LockAcquisitionError(f"Failed to acquire lock: {e}")
        
        # Failed to acquire
        logger.warning(
            f"Failed to acquire lock {self.lock_name} after {attempts} attempts"
        )
        raise LockAcquisitionError(
            f"Could not acquire lock {self.lock_name} after {attempts} attempts"
        )
    
    async def release(self) -> bool:
        """
        Release the lock.
        
        Returns:
            True if lock released
            
        Raises:
            LockReleaseError: If lock cannot be released
        """
        if not self.acquired:
            logger.warning(f"Lock {self.lock_name} not acquired, cannot release")
            return False
        
        if not self.lock_id:
            logger.error(f"Lock {self.lock_name} has no ID, cannot release safely")
            return False
        
        try:
            # Use Lua script for atomic release (only if we own it)
            if self.release_script_sha:
                result = await self.redis_client.evalsha(
                    self.release_script_sha,
                    1,
                    self.lock_name,
                    self.lock_id
                )
            else:
                # Fallback: non-atomic release
                current_value = await self.redis_client.get(self.lock_name)
                if current_value == self.lock_id:
                    result = await self.redis_client.delete(self.lock_name)
                else:
                    result = 0
            
            if result:
                logger.debug(f"Lock {self.lock_name} released (id={self.lock_id[:8]})")
                self.acquired = False
                self.lock_id = None
                return True
            else:
                logger.warning(
                    f"Lock {self.lock_name} was not released "
                    "(may have been acquired by another process)"
                )
                self.acquired = False
                self.lock_id = None
                return False
                
        except RedisError as e:
            logger.error(f"Redis error releasing lock {self.lock_name}: {e}")
            raise LockReleaseError(f"Failed to release lock: {e}")
    
    async def renew(self, additional_time: Optional[int] = None) -> bool:
        """
        Renew the lock timeout.
        
        Args:
            additional_time: Additional time in seconds (default: original timeout)
            
        Returns:
            True if renewed
        """
        if not self.acquired or not self.lock_id:
            logger.warning(f"Cannot renew lock {self.lock_name}: not acquired")
            return False
        
        try:
            timeout = additional_time or self.timeout
            
            # Use Lua script for atomic renewal
            if self.renew_script_sha:
                result = await self.redis_client.evalsha(
                    self.renew_script_sha,
                    1,
                    self.lock_name,
                    self.lock_id,
                    str(timeout)
                )
            else:
                # Fallback: check and renew separately (not atomic)
                current_value = await self.redis_client.get(self.lock_name)
                if current_value == self.lock_id:
                    result = await self.redis_client.expire(self.lock_name, timeout)
                else:
                    result = 0
            
            if result:
                logger.debug(f"Lock {self.lock_name} renewed for {timeout}s")
                return True
            else:
                logger.warning(f"Failed to renew lock {self.lock_name}")
                return False
                
        except RedisError as e:
            logger.error(f"Redis error renewing lock {self.lock_name}: {e}")
            return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.release()
        return False


class DistributedLockManager:
    """
    Manager for distributed locks.
    Handles lock creation and cleanup.
    """
    
    def __init__(self, redis_client: Redis):
        """
        Initialize lock manager.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis_client = redis_client
        self.locks: dict[str, DistributedLock] = {}
    
    def get_lock(
        self,
        lock_name: str,
        timeout: int = 30,
        retry_attempts: int = 3,
        retry_delay: float = 0.1
    ) -> DistributedLock:
        """
        Get or create a distributed lock.
        
        Args:
            lock_name: Name of the lock
            timeout: Lock timeout in seconds
            retry_attempts: Number of acquisition retry attempts
            retry_delay: Delay between retries
            
        Returns:
            DistributedLock instance
        """
        # Reuse existing lock if available
        if lock_name in self.locks:
            lock = self.locks[lock_name]
            if not lock.acquired:
                return lock
        
        # Create new lock
        lock = DistributedLock(
            redis_client=self.redis_client,
            lock_name=lock_name,
            timeout=timeout,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay
        )
        
        self.locks[lock_name] = lock
        return lock
    
    async def cleanup(self) -> None:
        """Release all acquired locks."""
        logger.info(f"Cleaning up {len(self.locks)} locks")
        
        for lock_name, lock in self.locks.items():
            if lock.acquired:
                try:
                    await lock.release()
                except Exception as e:
                    logger.error(f"Error releasing lock {lock_name}: {e}")
        
        self.locks.clear()


__all__ = [
    'DistributedLock',
    'DistributedLockManager',
    'LockAcquisitionError',
    'LockReleaseError'
]
```

---

## File 8: `backend/app/session/in_memory_session_store.py` (Fixed)

**Checklist:**
- [x] Fix race condition: return deep copies
- [x] Improve LRU eviction algorithm
- [x] Fix TTL preservation
- [x] Add proper timestamp handling
- [x] Implement atomic counter operations
- [x] Add comprehensive logging

```python
"""
In-memory session store implementation.
Suitable for development and single-instance deployments.

Version: 2.0.0 (Fixed race conditions and improved LRU)
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import OrderedDict
from copy import deepcopy

from .session_store import SessionStore
from .validators import SessionData

logger = logging.getLogger(__name__)


class InMemorySessionStore(SessionStore):
    """
    In-memory implementation of SessionStore.
    
    Features:
    - Thread-safe operations using asyncio locks
    - LRU eviction when max_sessions reached
    - TTL-based expiration
    - Atomic counter increments
    - Deep copy returns to prevent external mutations
    
    Limitations:
    - Sessions lost on restart
    - Not shared across multiple instances
    - Memory usage grows with number of sessions
    
    Fixed Issues:
    - Returns deep copies to prevent race conditions
    - Improved LRU eviction based on actual last_activity
    - TTL preservation on updates
    - Consistent timestamp handling
    """
    
    def __init__(
        self,
        max_sessions: int = 10000,
        default_ttl: int = 3600
    ):
        """
        Initialize in-memory session store.
        
        Args:
            max_sessions: Maximum number of sessions to keep
            default_ttl: Default TTL in seconds
        """
        self.sessions: OrderedDict[str, SessionData] = OrderedDict()
        self.expiry: Dict[str, datetime] = {}
        self.max_sessions = max_sessions
        self.default_ttl = default_ttl
        self.lock = asyncio.Lock()
        
        logger.info(
            f"InMemorySessionStore initialized "
            f"(max_sessions={max_sessions}, default_ttl={default_ttl}s)"
        )
    
    async def get(self, session_id: str) -> Optional[SessionData]:
        """
        Get session data by ID.
        
        Returns a deep copy to prevent external mutations.
        """
        async with self.lock:
            # Check if expired
            if session_id in self.expiry:
                if datetime.utcnow() > self.expiry[session_id]:
                    # Expired, remove it
                    del self.sessions[session_id]
                    del self.expiry[session_id]
                    logger.debug(f"Session {session_id} expired and removed")
                    return None
            
            # Get session data
            session_data = self.sessions.get(session_id)
            
            if session_data:
                # Update last activity
                session_data.last_activity = datetime.utcnow()
                
                # Move to end for LRU
                self.sessions.move_to_end(session_id)
                
                logger.debug(f"Retrieved session {session_id}")
                
                # CRITICAL FIX: Return deep copy to prevent race conditions
                return deepcopy(session_data)
            
            return None
    
    async def set(
        self,
        session_id: str,
        session_data: SessionData,
        ttl: Optional[int] = None
    ) -> bool:
        """Set session data with improved LRU eviction."""
        async with self.lock:
            now = datetime.utcnow()
            
            # Update existing session
            if session_id in self.sessions:
                # Preserve original created_at
                if not session_data.created_at:
                    existing_session = self.sessions[session_id]
                    session_data.created_at = existing_session.created_at or now
                
                session_data.updated_at = now
                session_data.last_activity = now
                
                # Store deep copy to prevent external mutations
                self.sessions[session_id] = deepcopy(session_data)
                self.sessions.move_to_end(session_id)
                
                # Update expiry
                ttl = ttl or self.default_ttl
                self.expiry[session_id] = now + timedelta(seconds=ttl)
                
                logger.debug(f"Updated existing session {session_id}")
                return True
            
            # Evict oldest session if at max capacity
            if len(self.sessions) >= self.max_sessions:
                # IMPROVED: Find truly oldest session by last_activity
                oldest_id = None
                oldest_activity = datetime.utcnow()
                
                for sid, sdata in list(self.sessions.items()):
                    if sdata.last_activity and sdata.last_activity < oldest_activity:
                        oldest_activity = sdata.last_activity
                        oldest_id = sid
                
                if oldest_id:
                    del self.sessions[oldest_id]
                    self.expiry.pop(oldest_id, None)
                    logger.info(
                        f"Evicted oldest session {oldest_id} "
                        f"(last activity: {oldest_activity})"
                    )
                else:
                    # Fallback to popitem if no last_activity found
                    oldest_id, _ = self.sessions.popitem(last=False)
                    self.expiry.pop(oldest_id, None)
                    logger.info(f"Evicted oldest session {oldest_id} (fallback)")
            
            # Add new session
            if not session_data.created_at:
                session_data.created_at = now
            session_data.updated_at = now
            session_data.last_activity = now
            
            # Store deep copy
            self.sessions[session_id] = deepcopy(session_data)
            self.sessions.move_to_end(session_id)
            
            # Set expiry
            ttl = ttl or self.default_ttl
            self.expiry[session_id] = now + timedelta(seconds=ttl)
            
            logger.debug(f"Created new session {session_id} (ttl={ttl}s)")
            return True
    
    async def update(
        self,
        session_id: str,
        updates: Dict[str, Any],
        atomic: bool = False
    ) -> bool:
        """Update session data with TTL preservation."""
        async with self.lock:
            session_data = self.sessions.get(session_id)
            
            if not session_data:
                logger.warning(f"Cannot update non-existent session {session_id}")
                return False
            
            # Get current TTL
            remaining_ttl = None
            if session_id in self.expiry:
                remaining_time = self.expiry[session_id] - datetime.utcnow()
                remaining_ttl = max(int(remaining_time.total_seconds()), 0)
            
            # Apply updates to a copy
            session_copy = deepcopy(session_data)
            
            for key, value in updates.items():
                if hasattr(session_copy, key):
                    setattr(session_copy, key, value)
                else:
                    # Store in metadata
                    session_copy.metadata[key] = value
            
            # Update timestamps
            session_copy.updated_at = datetime.utcnow()
            session_copy.last_activity = datetime.utcnow()
            
            # Store updated copy
            self.sessions[session_id] = session_copy
            
            # FIXED: Preserve TTL
            if remaining_ttl is not None:
                self.expiry[session_id] = datetime.utcnow() + timedelta(seconds=remaining_ttl)
            
            logger.debug(
                f"Updated session {session_id} (fields: {list(updates.keys())}, "
                f"remaining_ttl: {remaining_ttl}s)"
            )
            return True
    
    async def delete(self, session_id: str) -> bool:
        """Delete session data."""
        async with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                self.expiry.pop(session_id, None)
                logger.debug(f"Deleted session {session_id}")
                return True
            return False
    
    async def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        async with self.lock:
            # Check if expired
            if session_id in self.expiry:
                if datetime.utcnow() > self.expiry[session_id]:
                    return False
            
            return session_id in self.sessions
    
    async def list_active(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[str]:
        """List active session IDs."""
        async with self.lock:
            # Get non-expired sessions
            now = datetime.utcnow()
            active_sessions = [
                sid for sid in self.sessions.keys()
                if sid not in self.expiry or self.expiry[sid] > now
            ]
            
            # Apply pagination
            if limit:
                active_sessions = active_sessions[offset:offset + limit]
            else:
                active_sessions = active_sessions[offset:]
            
            return active_sessions
    
    async def count_active(self) -> int:
        """Count active sessions."""
        async with self.lock:
            now = datetime.utcnow()
            count = sum(
                1 for sid in self.sessions.keys()
                if sid not in self.expiry or self.expiry[sid] > now
            )
            return count
    
    async def cleanup_expired(self) -> int:
        """Clean up expired sessions."""
        async with self.lock:
            now = datetime.utcnow()
            expired_sessions = [
                sid for sid, expiry in self.expiry.items()
                if expiry <= now
            ]
            
            for session_id in expired_sessions:
                del self.sessions[session_id]
                del self.expiry[session_id]
            
            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            
            return len(expired_sessions)
    
    async def increment_counter(
        self,
        session_id: str,
        field: str,
        delta: int = 1
    ) -> int:
        """Atomically increment a counter field."""
        async with self.lock:
            session_data = self.sessions.get(session_id)
            
            if not session_data:
                logger.warning(
                    f"Cannot increment counter for non-existent session {session_id}"
                )
                return 0
            
            # Get current value
            if hasattr(session_data, field):
                current = getattr(session_data, field)
            else:
                current = session_data.metadata.get(field, 0)
            
            # Increment
            new_value = current + delta
            
            # Set new value
            if hasattr(session_data, field):
                setattr(session_data, field, new_value)
            else:
                session_data.metadata[field] = new_value
            
            # Update timestamp
            session_data.updated_at = datetime.utcnow()
            
            logger.debug(
                f"Incremented {field} for session {session_id}: "
                f"{current} -> {new_value}"
            )
            return new_value
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get session store statistics."""
        async with self.lock:
            now = datetime.utcnow()
            active_count = sum(
                1 for sid in self.sessions.keys()
                if sid not in self.expiry or self.expiry[sid] > now
            )
            
            return {
                "store_type": "in_memory",
                "total_sessions": len(self.sessions),
                "active_sessions": active_count,
                "expired_sessions": len(self.sessions) - active_count,
                "max_sessions": self.max_sessions,
                "utilization": f"{(len(self.sessions) / self.max_sessions * 100):.1f}%",
                "default_ttl": self.default_ttl
            }


__all__ = ['InMemorySessionStore']
```

This continues the implementation with critical race condition fixes. Should I continue with the Redis session store and remaining files?
