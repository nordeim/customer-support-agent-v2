"""
Redis-backed session store implementation.
Suitable for production multi-instance deployments.

Phase 4: Enables shared session state across multiple agent instances.
"""
import asyncio
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    import redis.asyncio as redis
    from redis.asyncio import Redis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    Redis = None
    RedisError = Exception
    RedisConnectionError = Exception
    REDIS_AVAILABLE = False

from .session_store import SessionStore, SessionData

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
    
    Requirements:
    - Redis 5.0+ (for Lua script support)
    - redis-py with asyncio support
    """
    
    # Lua script for atomic counter increment
    INCREMENT_SCRIPT = """
    local key = KEYS[1]
    local field = ARGV[1]
    local delta = tonumber(ARGV[2])
    local ttl = tonumber(ARGV[3])
    
    -- Get current session data
    local session_json = redis.call('GET', key)
    if not session_json then
        return nil
    end
    
    -- Parse JSON
    local session = cjson.decode(session_json)
    
    -- Increment field
    local current = tonumber(session[field]) or 0
    session[field] = current + delta
    
    -- Update timestamps
    session['updated_at'] = ARGV[4]
    session['last_activity'] = ARGV[4]
    
    -- Save back to Redis
    redis.call('SET', key, cjson.encode(session), 'EX', ttl)
    
    return session[field]
    """
    
    def __init__(
        self,
        redis_url: str,
        key_prefix: str = "session:",
        default_ttl: int = 3600,
        max_connections: int = 10
    ):
        """
        Initialize Redis session store.
        
        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for session keys
            default_ttl: Default TTL in seconds
            max_connections: Maximum connection pool size
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
        
        # Create connection pool
        self.pool = redis.ConnectionPool.from_url(
            redis_url,
            max_connections=max_connections,
            decode_responses=True
        )
        
        self.client: Optional[Redis] = None
        self.increment_script_sha: Optional[str] = None
        
        logger.info(
            f"RedisSessionStore initialized "
            f"(url={redis_url}, prefix={key_prefix}, ttl={default_ttl}s)"
        )
    
    async def _ensure_connection(self) -> Redis:
        """
        Ensure Redis connection is established.
        
        Returns:
            Redis client
        """
        if self.client is None:
            self.client = Redis(connection_pool=self.pool)
            
            # Load Lua script
            try:
                self.increment_script_sha = await self.client.script_load(
                    self.INCREMENT_SCRIPT
                )
                logger.info("Loaded Lua increment script into Redis")
            except RedisError as e:
                logger.error(f"Failed to load Lua script: {e}")
        
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
    
    async def get(self, session_id: str) -> Optional[SessionData]:
        """Get session data by ID."""
        try:
            client = await self._ensure_connection()
            key = self._make_key(session_id)
            
            # Get session JSON from Redis
            session_json = await client.get(key)
            
            if not session_json:
                return None
            
            # Parse JSON to SessionData
            session_data = SessionData.from_json(session_json)
            
            # Update last activity timestamp
            session_data.last_activity = datetime.utcnow()
            
            # Persist updated timestamp
            await self.set(session_id, session_data)
            
            logger.debug(f"Retrieved session {session_id} from Redis")
            return session_data
            
        except RedisError as e:
            logger.error(f"Redis error getting session {session_id}: {e}")
            return None
    
    async def set(
        self,
        session_id: str,
        session_data: SessionData,
        ttl: Optional[int] = None
    ) -> bool:
        """Set session data."""
        try:
            client = await self._ensure_connection()
            key = self._make_key(session_id)
            
            # Set timestamps
            now = datetime.utcnow()
            if not session_data.created_at:
                session_data.created_at = now
            session_data.updated_at = now
            if not session_data.last_activity:
                session_data.last_activity = now
            
            # Serialize to JSON
            session_json = session_data.to_json()
            
            # Set in Redis with TTL
            ttl = ttl or self.default_ttl
            await client.set(key, session_json, ex=ttl)
            
            logger.debug(f"Set session {session_id} in Redis (ttl={ttl}s)")
            return True
            
        except RedisError as e:
            logger.error(f"Redis error setting session {session_id}: {e}")
            return False
    
    async def update(
        self,
        session_id: str,
        updates: Dict[str, Any],
        atomic: bool = False
    ) -> bool:
        """Update session data."""
        try:
            client = await self._ensure_connection()
            key = self._make_key(session_id)
            
            # Get current session data
            session_json = await client.get(key)
            if not session_json:
                logger.warning(f"Cannot update non-existent session {session_id}")
                return False
            
            # Parse session data
            session_data = SessionData.from_json(session_json)
            
            # Apply updates
            for field, value in updates.items():
                if hasattr(session_data, field):
                    setattr(session_data, field, value)
                else:
                    session_data.metadata[field] = value
            
            # Update timestamps
            session_data.updated_at = datetime.utcnow()
            session_data.last_activity = datetime.utcnow()
            
            # Save back to Redis
            await self.set(session_id, session_data)
            
            logger.debug(f"Updated session {session_id} in Redis (fields: {list(updates.keys())})")
            return True
            
        except RedisError as e:
            logger.error(f"Redis error updating session {session_id}: {e}")
            return False
    
    async def delete(self, session_id: str) -> bool:
        """Delete session data."""
        try:
            client = await self._ensure_connection()
            key = self._make_key(session_id)
            
            result = await client.delete(key)
            
            if result > 0:
                logger.debug(f"Deleted session {session_id} from Redis")
                return True
            return False
            
        except RedisError as e:
            logger.error(f"Redis error deleting session {session_id}: {e}")
            return False
    
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
    
    async def list_active(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[str]:
        """List active session IDs."""
        try:
            client = await self._ensure_connection()
            pattern = f"{self.key_prefix}*"
            
            # Use SCAN to iterate through keys
            session_ids = []
            cursor = 0
            
            while True:
                cursor, keys = await client.scan(
                    cursor,
                    match=pattern,
                    count=100
                )
                
                # Extract session IDs from keys
                for key in keys:
                    session_id = key.replace(self.key_prefix, '', 1)
                    session_ids.append(session_id)
                
                if cursor == 0:
                    break
            
            # Apply pagination
            if limit:
                session_ids = session_ids[offset:offset + limit]
            else:
                session_ids = session_ids[offset:]
            
            return session_ids
            
        except RedisError as e:
            logger.error(f"Redis error listing sessions: {e}")
            return []
    
    async def count_active(self) -> int:
        """Count active sessions."""
        try:
            client = await self._ensure_connection()
            pattern = f"{self.key_prefix}*"
            
            # Count keys matching pattern
            count = 0
            cursor = 0
            
            while True:
                cursor, keys = await client.scan(
                    cursor,
                    match=pattern,
                    count=100
                )
                count += len(keys)
                
                if cursor == 0:
                    break
            
            return count
            
        except RedisError as e:
            logger.error(f"Redis error counting sessions: {e}")
            return 0
    
    async def cleanup_expired(self) -> int:
        """
        Clean up expired sessions.
        
        Note: Redis automatically removes expired keys,
        so this is a no-op for Redis store.
        
        Returns:
            0 (Redis handles expiration automatically)
        """
        logger.debug("Redis handles expiration automatically")
        return 0
    
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
            
            # Get current TTL to preserve it
            ttl = await client.ttl(key)
            if ttl < 0:
                ttl = self.default_ttl
            
            # Execute Lua script for atomic increment
            now_iso = datetime.utcnow().isoformat()
            
            result = await client.evalsha(
                self.increment_script_sha,
                1,  # Number of keys
                key,  # KEYS[1]
                field,  # ARGV[1]
                str(delta),  # ARGV[2]
                str(ttl),  # ARGV[3]
                now_iso  # ARGV[4]
            )
            
            if result is None:
                logger.error(f"Lua script returned nil for session {session_id}")
                return 0
            
            new_value = int(result)
            logger.debug(f"Atomically incremented {field} for session {session_id} to {new_value}")
            return new_value
            
        except RedisError as e:
            logger.error(f"Redis error incrementing counter for session {session_id}: {e}")
            
            # Fallback to non-atomic increment
            logger.warning("Falling back to non-atomic increment")
            session_data = await self.get(session_id)
            if session_data:
                current = getattr(session_data, field, 0)
                new_value = current + delta
                await self.update(session_id, {field: new_value})
                return new_value
            
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get session store statistics."""
        try:
            client = await self._ensure_connection()
            
            # Get Redis info
            info = await client.info('stats')
            memory_info = await client.info('memory')
            
            # Count sessions
            active_count = await self.count_active()
            
            return {
                "store_type": "redis",
                "active_sessions": active_count,
                "redis_version": info.get('redis_version', 'unknown'),
                "connected_clients": info.get('connected_clients', 0),
                "used_memory_human": memory_info.get('used_memory_human', 'unknown'),
                "total_commands_processed": info.get('total_commands_processed', 0)
            }
            
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
            logger.info("Closed Redis connection")


# Export
__all__ = ['RedisSessionStore', 'REDIS_AVAILABLE']
