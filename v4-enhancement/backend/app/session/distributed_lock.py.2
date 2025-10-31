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
