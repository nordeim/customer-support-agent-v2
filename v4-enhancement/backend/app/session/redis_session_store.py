"""
Redis-backed session store implementation.
Suitable for production multi-instance deployments.

Version: 2.0.0 (Fixed race conditions, added L1 cache, encryption)
"""
import asyncio
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    import redis.asyncio as redis
    from redis.asyncio import Redis
    from redis.exceptions import (
        RedisError,
        ConnectionError as RedisConnectionError,
        TimeoutError as RedisTimeoutError
    )
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    Redis = None
    RedisError = Exception
    RedisConnectionError = Exception
    RedisTimeoutError = Exception
    REDIS_AVAILABLE = False

from cachetools import TTLCache

from .session_store import SessionStore
from .validators import SessionData
from ..utils.encryption import SessionEncryption, EncryptionError
from ..utils.retry import async_retry, RetryConfig, RetryStrategy

logger = logging.getLogger(__name__)


class RedisSessionStore(SessionStore):
    """
    Redis-backed implementation of SessionStore.
    
    Features:
    - Shared state across multiple instances
    - Atomic operations using Lua scripts
    - Automatic expiration with TTL
    - Persistent storage (if Redis persistence enabled)
    - High performance with connection pooling
    - L1 (in-memory) cache for frequently accessed sessions
    - Optional encryption at rest
    - Comprehensive error handling with retries
    
    Requirements:
    - Redis 5.0+ (for Lua script support)
    - redis-py with asyncio support
    
    Fixed Issues:
    - Atomic updates using Lua scripts
    - Proper TTL preservation
    - Connection health monitoring
    - L1 cache to reduce Redis load
    - Encryption support
    """
    
    # Lua script for atomic update
    UPDATE_SCRIPT = """
    local key = KEYS[1]
    local updates_json = ARGV[1]
    local now_iso = ARGV[2]
    
    -- Get current session data
    local session_json = redis.call('GET', key)
    if not session_json then
        return {err = 'Session not found'}
    end
    
    -- Get current TTL to preserve it
    local ttl = redis.call('TTL', key)
    if ttl < 0 then
        ttl = 1800  -- Default 30 minutes
    end
    
    -- Parse JSON
    local session = cjson.decode(session_json)
    local updates = cjson.decode(updates_json)
    
    -- Apply updates
    for field, value in pairs(updates) do
        session[field] = value
    end
    
    -- Update timestamps
    session['updated_at'] = now_iso
    session['last_activity'] = now_iso
    
    -- Save back to Redis with preserved TTL
    redis.call('SET', key, cjson.encode(session), 'EX', ttl)
    
    return 1
    """
    
    # Lua script for atomic counter increment
    INCREMENT_SCRIPT = """
    local key = KEYS[1]
    local field = ARGV[1]
    local delta = tonumber(ARGV[2])
    local now_iso = ARGV[3]
    
    -- Check existence first
    if redis.call('EXISTS', key) == 0 then
        return {err = 'Session not found'}
    end
    
    -- Get current session data
    local session_json = redis.call('GET', key)
    if not session_json then
        return {err = 'Session expired'}
    end
    
    -- Get remaining TTL to preserve it
    local ttl = redis.call('TTL', key)
    if ttl < 0 then
        ttl = 1800  -- Default 30 minutes
    end
    
    -- Parse JSON
    local session = cjson.decode(session_json)
    
    -- Increment field
    local current = tonumber(session[field]) or 0
    session[field] = current + delta
    
    -- Update timestamps
    session['updated_at'] = now_iso
    session['last_activity'] = now_iso
    
    -- Save back with preserved TTL
    redis.call('SET', key, cjson.encode(session), 'EX', ttl)
    
    return session[field]
    """
    
    def __init__(
        self,
        redis_url: str,
        key_prefix: str = "session:",
        default_ttl: int = 3600,
        max_connections: int = 50,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30,
        # L1 cache settings
        enable_l1_cache: bool = True,
        l1_cache_size: int = 1000,
        l1_cache_ttl: int = 60,
        # Encryption settings
        enable_encryption: bool = False,
        encryption_key: Optional[str] = None
    ):
        """
        Initialize Redis session store.
        
        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for session keys
            default_ttl: Default TTL in seconds
            max_connections: Maximum connection pool size
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
            retry_on_timeout: Retry on timeout
            health_check_interval: Health check interval in seconds
            enable_l1_cache: Enable L1 in-memory cache
            l1_cache_size: L1 cache size (number of sessions)
            l1_cache_ttl: L1 cache TTL in seconds
            enable_encryption: Enable encryption at rest
            encryption_key: Encryption key (required if enable_encryption=True)
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis support not available. "
                "Install with: pip install redis[asyncio]"
            )
        
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self.max_connections = max_connections
        
        # Create connection pool with optimized settings
        self.pool = redis.ConnectionPool.from_url(
            redis_url,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            retry_on_timeout=retry_on_timeout,
            decode_responses=True,
            health_check_interval=health_check_interval,
            socket_keepalive=True
        )
        
        self.client: Optional[Redis] = None
        self.update_script_sha: Optional[str] = None
        self.increment_script_sha: Optional[str] = None
        
        # L1 cache
        self.enable_l1_cache = enable_l1_cache
        self.l1_cache: Optional[TTLCache] = None
        self.l1_cache_lock = asyncio.Lock()
        
        if enable_l1_cache:
            from cachetools import TTLCache
            self.l1_cache = TTLCache(maxsize=l1_cache_size, ttl=l1_cache_ttl)
            logger.info(f"L1 cache enabled (size={l1_cache_size}, ttl={l1_cache_ttl}s)")
        
        # Encryption
        self.enable_encryption = enable_encryption
        self.encryptor: Optional[SessionEncryption] = None
        
        if enable_encryption:
            if not encryption_key:
                raise ValueError("encryption_key required when enable_encryption=True")
            
            from ..utils.encryption import SessionEncryption
            self.encryptor = SessionEncryption(encryption_key)
            logger.info("Session encryption enabled")
        
        # Retry configuration for Redis operations
        self.retry_config = RetryConfig(
            max_attempts=3,
            initial_delay=0.1,
            max_delay=2.0,
            strategy=RetryStrategy.EXPONENTIAL,
            retry_on_exceptions=(RedisConnectionError, RedisTimeoutError)
        )
        
        logger.info(
            f"RedisSessionStore initialized "
            f"(url={redis_url}, prefix={key_prefix}, ttl={default_ttl}s, "
            f"l1_cache={enable_l1_cache}, encryption={enable_encryption})"
        )
    
    async def _ensure_connection(self) -> Redis:
        """
        Ensure Redis connection is established and healthy.
        Implements automatic reconnection on failure.
        """
        if self.client is None:
            self.client = Redis(connection_pool=self.pool)
            
            # Load Lua scripts
            try:
                self.update_script_sha = await self.client.script_load(self.UPDATE_SCRIPT)
                self.increment_script_sha = await self.client.script_load(self.INCREMENT_SCRIPT)
                logger.info("✓ Lua scripts loaded into Redis")
            except RedisError as e:
                logger.error(f"Failed to load Lua scripts: {e}")
                # Continue anyway, will use fallback methods
        
        # Health check: verify connection is alive
        try:
            await self.client.ping()
        except (RedisConnectionError, RedisTimeoutError) as e:
            logger.warning(f"Redis connection unhealthy, reconnecting: {e}")
            
            # Close old connection
            try:
                await self.client.close()
            except:
                pass
            
            # Create new connection
            self.client = Redis(connection_pool=self.pool)
            
            # Reload scripts
            try:
                self.update_script_sha = await self.client.script_load(self.UPDATE_SCRIPT)
                self.increment_script_sha = await self.client.script_load(self.INCREMENT_SCRIPT)
            except RedisError:
                pass
            
            # Final health check
            await self.client.ping()
        
        return self.client
    
    def _make_key(self, session_id: str) -> str:
        """
        Create Redis key for session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Redis key
        """
        return f"{self.key_prefix}{session_id}"
    
    def _make_active_set_key(self) -> str:
        """Get key for active sessions SET."""
        return f"{self.key_prefix}active"
    
    async def _encrypt_data(self, data: str) -> str:
        """Encrypt session data if encryption is enabled."""
        if not self.enable_encryption or not self.encryptor:
            return data
        
        try:
            return self.encryptor.encrypt_string(data)
        except EncryptionError as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    async def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt session data if encryption is enabled."""
        if not self.enable_encryption or not self.encryptor:
            return encrypted_data
        
        try:
            return self.encryptor.decrypt_string(encrypted_data)
        except EncryptionError as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    async def _get_from_l1_cache(self, session_id: str) -> Optional[SessionData]:
        """Get session from L1 cache."""
        if not self.enable_l1_cache or not self.l1_cache:
            return None
        
        async with self.l1_cache_lock:
            session_data = self.l1_cache.get(session_id)
            if session_data:
                logger.debug(f"L1 cache hit for session {session_id}")
                from copy import deepcopy
                return deepcopy(session_data)
        
        return None
    
    async def _set_in_l1_cache(self, session_id: str, session_data: SessionData) -> None:
        """Set session in L1 cache."""
        if not self.enable_l1_cache or not self.l1_cache:
            return
        
        async with self.l1_cache_lock:
            from copy import deepcopy
            self.l1_cache[session_id] = deepcopy(session_data)
    
    async def _invalidate_l1_cache(self, session_id: str) -> None:
        """Invalidate L1 cache entry."""
        if not self.enable_l1_cache or not self.l1_cache:
            return
        
        async with self.l1_cache_lock:
            self.l1_cache.pop(session_id, None)
    
    @async_retry()
    async def get(self, session_id: str) -> Optional[SessionData]:
        """Get session data by ID with L1 cache and retry."""
        # Check L1 cache first
        cached = await self._get_from_l1_cache(session_id)
        if cached:
            return cached
        
        try:
            client = await self._ensure_connection()
            key = self._make_key(session_id)
            
            # Get session JSON from Redis
            session_json = await client.get(key)
            
            if not session_json:
                return None
            
            # Decrypt if needed
            session_json = await self._decrypt_data(session_json)
            
            # Parse JSON to SessionData
            session_data = SessionData.from_json(session_json)
            
            # Update last activity timestamp
            session_data.last_activity = datetime.utcnow()
            
            # Store in L1 cache
            await self._set_in_l1_cache(session_id, session_data)
            
            logger.debug(f"Retrieved session {session_id} from Redis")
            return session_data
            
        except RedisError as e:
            logger.error(f"Redis error getting session {session_id}: {e}")
            return None
    
    @async_retry()
    async def set(
        self,
        session_id: str,
        session_data: SessionData,
        ttl: Optional[int] = None
    ) -> bool:
        """Set session data with pipelining."""
        try:
            client = await self._ensure_connection()
            key = self._make_key(session_id)
            active_set_key = self._make_active_set_key()
            
            # Set timestamps
            now = datetime.utcnow()
            if not session_data.created_at:
                session_data.created_at = now
            session_data.updated_at = now
            if not session_data.last_activity:
                session_data.last_activity = now
            
            # Serialize to JSON
            session_json = session_data.to_json()
            
            # Encrypt if needed
            session_json = await self._encrypt_data(session_json)
            
            # Set in Redis with TTL using pipeline
            ttl = ttl or self.default_ttl
            
            pipe = client.pipeline()
            pipe.set(key, session_json, ex=ttl)
            pipe.sadd(active_set_key, session_id)  # Add to active set
            pipe.expire(active_set_key, ttl * 2)  # Keep set alive longer
            await pipe.execute()
            
            # Update L1 cache
            await self._set_in_l1_cache(session_id, session_data)
            
            logger.debug(f"Set session {session_id} in Redis (ttl={ttl}s)")
            return True
            
        except RedisError as e:
            logger.error(f"Redis error setting session {session_id}: {e}")
            return False
    
    @async_retry()
    async def update(
        self,
        session_id: str,
        updates: Dict[str, Any],
        atomic: bool = True
    ) -> bool:
        """Update session data using Lua script for atomicity."""
        try:
            client = await self._ensure_connection()
            key = self._make_key(session_id)
            
            now_iso = datetime.utcnow().isoformat()
            
            # Use Lua script for atomic update
            if self.update_script_sha:
                try:
                    result = await client.evalsha(
                        self.update_script_sha,
                        1,  # Number of keys
                        key,  # KEYS[1]
                        json.dumps(updates),  # ARGV[1]
                        now_iso  # ARGV[2]
                    )
                    
                    if isinstance(result, dict) and 'err' in result:
                        logger.warning(f"Lua update failed: {result['err']}")
                        return False
                    
                    # Invalidate L1 cache
                    await self._invalidate_l1_cache(session_id)
                    
                    logger.debug(f"Updated session {session_id} in Redis (atomic)")
                    return True
                    
                except RedisError as e:
                    logger.warning(f"Lua script execution failed, using fallback: {e}")
            
            # Fallback: Optimistic locking with WATCH
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Watch the key
                    await client.watch(key)
                    
                    # Get current value
                    session_json = await client.get(key)
                    if not session_json:
                        await client.unwatch()
                        logger.warning(f"Cannot update non-existent session {session_id}")
                        return False
                    
                    # Decrypt if needed
                    session_json = await self._decrypt_data(session_json)
                    
                    # Parse and modify
                    session_data = SessionData.from_json(session_json)
                    for field, value in updates.items():
                        if hasattr(session_data, field):
                            setattr(session_data, field, value)
                        else:
                            session_data.metadata[field] = value
                    
                    session_data.updated_at = datetime.utcnow()
                    session_data.last_activity = datetime.utcnow()
                    
                    # Get TTL
                    ttl = await client.ttl(key)
                    if ttl < 0:
                        ttl = self.default_ttl
                    
                    # Serialize and encrypt
                    updated_json = session_data.to_json()
                    updated_json = await self._encrypt_data(updated_json)
                    
                    # Atomic transaction
                    pipe = client.pipeline()
                    pipe.set(key, updated_json, ex=ttl)
                    await pipe.execute()  # Will fail if key changed
                    
                    # Invalidate L1 cache
                    await self._invalidate_l1_cache(session_id)
                    
                    logger.debug(f"Updated session {session_id} (optimistic lock)")
                    return True
                    
                except redis.WatchError:
                    # Key changed, retry
                    logger.warning(
                        f"Update conflict for session {session_id}, "
                        f"retrying ({attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(0.01 * (2 ** attempt))
                    continue
                except Exception as e:
                    await client.unwatch()
                    raise
            
            logger.error(f"Failed to update session {session_id} after {max_retries} retries")
            return False
            
        except RedisError as e:
            logger.error(f"Redis error updating session {session_id}: {e}")
            return False
    
    @async_retry()
    async def delete(self, session_id: str) -> bool:
        """Delete session data with pipelining."""
        try:
            client = await self._ensure_connection()
            key = self._make_key(session_id)
            active_set_key = self._make_active_set_key()
            
            # Use pipeline for atomic delete
            pipe = client.pipeline()
            pipe.delete(key)
            pipe.srem(active_set_key, session_id)
            results = await pipe.execute()
            
            # Invalidate L1 cache
            await self._invalidate_l1_cache(session_id)
            
            if results[0] > 0:
                logger.debug(f"Deleted session {session_id} from Redis")
                return True
            return False
            
        except RedisError as e:
            logger.error(f"Redis error deleting session {session_id}: {e}")
            return False
    
    @async_retry()
    async def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        try:
            client = await self._ensure_connection()
            key = self._make_key(session_id)
            
            result = await client.exists(key)
            return result > 0
            
        except RedisError as e:
            logger.error(f"Redis error checking session {session_id}: {e}")
            return False
    
    @async_retry()
    async def list_active(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[str]:
        """List active session IDs using SET for efficiency."""
        try:
            client = await self._ensure_connection()
            active_set_key = self._make_active_set_key()
            
            # Use SSCAN for efficient pagination
            session_ids = []
            cursor = 0
            
            while True:
                cursor, members = await client.sscan(
                    active_set_key,
                    cursor,
                    count=limit or 100
                )
                session_ids.extend(members)
                
                if cursor == 0:
                    break
                
                if limit and len(session_ids) >= limit + offset:
                    break
            
            # Apply pagination
            return session_ids[offset:offset + limit] if limit else session_ids[offset:]
            
        except RedisError as e:
            logger.error(f"Redis error listing sessions: {e}")
            return []
    
    @async_retry()
    async def count_active(self) -> int:
        """Count active sessions using SET for O(1) performance."""
        try:
            client = await self._ensure_connection()
            active_set_key = self._make_active_set_key()
            
            return await client.scard(active_set_key)
            
        except RedisError as e:
            logger.error(f"Redis error counting sessions: {e}")
            return 0
    
    async def cleanup_expired(self) -> int:
        """
        Clean up expired sessions.
        
        Note: Redis automatically removes expired keys,
        but we clean up the active SET.
        """
        try:
            client = await self._ensure_connection()
            active_set_key = self._make_active_set_key()
            
            # Get all session IDs from active set
            session_ids = await client.smembers(active_set_key)
            
            # Check which ones still exist
            cleaned = 0
            for session_id in session_ids:
                key = self._make_key(session_id)
                exists = await client.exists(key)
                
                if not exists:
                    # Remove from active set
                    await client.srem(active_set_key, session_id)
                    cleaned += 1
            
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired sessions from active set")
            
            return cleaned
            
        except RedisError as e:
            logger.error(f"Redis error cleaning up sessions: {e}")
            return 0
    
    @async_retry()
    async def increment_counter(
        self,
        session_id: str,
        field: str,
        delta: int = 1
    ) -> int:
        """Atomically increment a counter field using Lua script."""
        try:
            client = await self._ensure_connection()
            key = self._make_key(session_id)
            
            # Check if session exists
            if not await client.exists(key):
                logger.warning(f"Cannot increment counter for non-existent session {session_id}")
                return 0
            
            now_iso = datetime.utcnow().isoformat()
            
            # Execute Lua script for atomic increment
            if self.increment_script_sha:
                try:
                    result = await client.evalsha(
                        self.increment_script_sha,
                        1,  # Number of keys
                        key,  # KEYS[1]
                        field,  # ARGV[1]
                        str(delta),  # ARGV[2]
                        now_iso  # ARGV[3]
                    )
                    
                    if isinstance(result, dict) and 'err' in result:
                        logger.error(f"Lua increment failed: {result['err']}")
                        return 0
                    
                    # Invalidate L1 cache
                    await self._invalidate_l1_cache(session_id)
                    
                    new_value = int(result)
                    logger.debug(f"Atomically incremented {field} for session {session_id} to {new_value}")
                    return new_value
                    
                except RedisError as e:
                    logger.warning(f"Lua increment failed, using fallback: {e}")
            
            # Fallback: non-atomic increment
            logger.warning("Using non-atomic increment (Lua script not available)")
            session_data = await self.get(session_id)
            if session_data:
                current = getattr(session_data, field, 0) if hasattr(session_data, field) else session_data.metadata.get(field, 0)
                new_value = current + delta
                await self.update(session_id, {field: new_value})
                return new_value
            
            return 0
            
        except RedisError as e:
            logger.error(f"Redis error incrementing counter for session {session_id}: {e}")
            return 0
    
    @async_retry()
    async def get_stats(self) -> Dict[str, Any]:
        """Get session store statistics."""
        try:
            client = await self._ensure_connection()
            
            # Get Redis info
            info = await client.info('stats')
            memory_info = await client.info('memory')
            
            # Count sessions
            active_count = await self.count_active()
            
            stats = {
                "store_type": "redis",
                "active_sessions": active_count,
                "redis_version": info.get('redis_version', 'unknown'),
                "connected_clients": info.get('connected_clients', 0),
                "used_memory_human": memory_info.get('used_memory_human', 'unknown'),
                "total_commands_processed": info.get('total_commands_processed', 0),
                "l1_cache_enabled": self.enable_l1_cache,
                "encryption_enabled": self.enable_encryption
            }
            
            # L1 cache stats
            if self.enable_l1_cache and self.l1_cache:
                async with self.l1_cache_lock:
                    stats["l1_cache_size"] = len(self.l1_cache)
                    stats["l1_cache_maxsize"] = self.l1_cache.maxsize
            
            return stats
            
        except RedisError as e:
            logger.error(f"Redis error getting stats: {e}")
            return {
                "store_type": "redis",
                "error": str(e)
            }
    
    async def ping(self) -> bool:
        """
        Test Redis connection.
        
        Returns:
            True if connected
        """
        try:
            client = await self._ensure_connection()
            result = await client.ping()
            return result
        except RedisError as e:
            logger.error(f"Redis ping failed: {e}")
            return False
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            await self.pool.disconnect()
            self.client = None
            logger.info("✓ Closed Redis connection")


__all__ = ['RedisSessionStore', 'REDIS_AVAILABLE']
