# Phase 4 Implementation: Session State Externalization (Redis) ✓

I will now create **complete, production-ready files** for Phase 4 with Redis-backed session storage for multi-instance deployments.

---

## File 1: `backend/app/session/__init__.py` (NEW)

```python
"""
Session management package.
Provides session storage abstractions for agent context persistence.

Phase 4: Enables horizontal scaling with shared session state.
"""
from .session_store import SessionStore, SessionData
from .in_memory_session_store import InMemorySessionStore

# Conditionally import Redis store
try:
    from .redis_session_store import RedisSessionStore
    REDIS_AVAILABLE = True
except ImportError:
    RedisSessionStore = None
    REDIS_AVAILABLE = False

__all__ = [
    'SessionStore',
    'SessionData',
    'InMemorySessionStore',
    'RedisSessionStore',
    'REDIS_AVAILABLE'
]
```

---

## File 2: `backend/app/session/session_store.py` (NEW)

```python
"""
Abstract session store interface.
Defines the contract for session persistence implementations.

Phase 4: Enables pluggable session storage backends.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json


@dataclass
class SessionData:
    """
    Session data structure for persistence.
    Represents agent context that needs to survive across instances.
    """
    session_id: str
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    message_count: int = 0
    escalated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        data = asdict(self)
        
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
        # Convert ISO format strings back to datetime
        for key in ['created_at', 'updated_at', 'last_activity']:
            if key in data and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])
        
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


class SessionStore(ABC):
    """
    Abstract base class for session storage.
    
    Implementations must provide thread-safe operations for:
    - Getting session data
    - Setting session data
    - Updating session data (with atomic operations for counters)
    - Deleting session data
    - Listing active sessions
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
            atomic: Whether to use atomic operations for counters
            
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


# Export public API
__all__ = ['SessionStore', 'SessionData']
```

---

## File 3: `backend/app/session/in_memory_session_store.py` (NEW)

```python
"""
In-memory session store implementation.
Suitable for development and single-instance deployments.

Phase 4: Provides local session storage without external dependencies.
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import OrderedDict

from .session_store import SessionStore, SessionData

logger = logging.getLogger(__name__)


class InMemorySessionStore(SessionStore):
    """
    In-memory implementation of SessionStore.
    
    Features:
    - Thread-safe operations using asyncio locks
    - LRU eviction when max_sessions reached
    - TTL-based expiration
    - Atomic counter increments
    
    Limitations:
    - Sessions lost on restart
    - Not shared across multiple instances
    - Memory usage grows with number of sessions
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
        """Get session data by ID."""
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
                # Move to end (LRU)
                self.sessions.move_to_end(session_id)
                logger.debug(f"Retrieved session {session_id}")
            
            return session_data
    
    async def set(
        self,
        session_id: str,
        session_data: SessionData,
        ttl: Optional[int] = None
    ) -> bool:
        """Set session data."""
        async with self.lock:
            # Evict oldest session if at max capacity
            if len(self.sessions) >= self.max_sessions and session_id not in self.sessions:
                oldest_id, _ = self.sessions.popitem(last=False)
                self.expiry.pop(oldest_id, None)
                logger.info(f"Evicted oldest session {oldest_id} (max capacity reached)")
            
            # Set timestamps
            now = datetime.utcnow()
            if not session_data.created_at:
                session_data.created_at = now
            session_data.updated_at = now
            session_data.last_activity = now
            
            # Store session
            self.sessions[session_id] = session_data
            self.sessions.move_to_end(session_id)
            
            # Set expiry
            ttl = ttl or self.default_ttl
            self.expiry[session_id] = now + timedelta(seconds=ttl)
            
            logger.debug(f"Set session {session_id} (ttl={ttl}s)")
            return True
    
    async def update(
        self,
        session_id: str,
        updates: Dict[str, Any],
        atomic: bool = False
    ) -> bool:
        """Update session data."""
        async with self.lock:
            session_data = self.sessions.get(session_id)
            
            if not session_data:
                logger.warning(f"Cannot update non-existent session {session_id}")
                return False
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(session_data, key):
                    setattr(session_data, key, value)
                else:
                    # Store in metadata
                    session_data.metadata[key] = value
            
            # Update timestamp
            session_data.updated_at = datetime.utcnow()
            session_data.last_activity = datetime.utcnow()
            
            logger.debug(f"Updated session {session_id} (fields: {list(updates.keys())})")
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
                logger.warning(f"Cannot increment counter for non-existent session {session_id}")
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
            
            logger.debug(f"Incremented {field} for session {session_id}: {current} -> {new_value}")
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
                "utilization": f"{(len(self.sessions) / self.max_sessions * 100):.1f}%"
            }


# Export
__all__ = ['InMemorySessionStore']
```

---

## File 4: `backend/app/session/redis_session_store.py` (NEW)

```python
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
```

---

## File 5: `backend/app/agents/chat_agent.py` (UPDATED - Complete Replacement with SessionStore)

```python
"""
Customer Support Agent implementation with full tool integration.
This agent orchestrates RAG, Memory, Attachment, and Escalation tools.

Phase 4 Update: Uses SessionStore for externalized session state.
"""
import asyncio
import json
import logging
import uuid
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass, field

from ..config import settings
from ..config.tool_settings import tool_settings
from ..tools.base_tool import ToolResult
from ..tools.tool_call_wrapper import (
    call_tool_with_wrapper,
    tool_call_context,
    RetryConfig,
    get_tool_metrics
)
from ..session import SessionStore, SessionData, InMemorySessionStore
from ..models.session import Session
from ..models.message import Message
from ..models.memory import Memory

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """
    Context for agent processing (DEPRECATED in Phase 4).
    Use SessionData from SessionStore instead.
    Kept for backward compatibility.
    """
    session_id: str
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    message_count: int = 0
    escalated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None


class AgentResponse:
    """Structured agent response."""
    
    def __init__(
        self,
        message: str,
        sources: List[Dict] = None,
        requires_escalation: bool = False,
        confidence: float = 0.0,
        tools_used: List[str] = None,
        processing_time: float = 0.0
    ):
        self.message = message
        self.sources = sources or []
        self.requires_escalation = requires_escalation
        self.confidence = confidence
        self.tools_used = tools_used or []
        self.processing_time = processing_time
        self.tool_metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message,
            "sources": self.sources,
            "requires_escalation": self.requires_escalation,
            "confidence": self.confidence,
            "tools_used": self.tools_used,
            "processing_time": self.processing_time,
            "metadata": self.tool_metadata
        }


class CustomerSupportAgent:
    """
    Production-ready customer support agent with full tool integration.
    Orchestrates multiple AI tools for comprehensive support capabilities.
    
    Phase 4: Uses SessionStore for shared state across instances.
    """
    
    # System prompt with tool instructions
    SYSTEM_PROMPT = """You are an expert customer support AI assistant with access to the following tools:

AVAILABLE TOOLS:
1. **rag_search**: Search our knowledge base for relevant information
   - Use this when users ask questions about policies, procedures, or general information
   - Always cite sources when using information from this tool

2. **memory_management**: Store and retrieve conversation context
   - Use this to remember important user information and preferences
   - Check memory at the start of each conversation for context

3. **attachment_processor**: Process and analyze uploaded documents
   - Use this when users upload files
   - Extract and analyze content from various file formats

4. **escalation_check**: Determine if human intervention is needed
   - Monitor for signs that require human support
   - Check sentiment and urgency of user messages

INSTRUCTIONS:
1. Always be helpful, professional, and empathetic
2. Use tools appropriately to provide accurate information
3. Cite your sources when providing information from the knowledge base
4. Remember important details about the user and their issues
5. Escalate to human support when:
   - The user explicitly asks for human assistance
   - The issue involves legal or compliance matters
   - The user expresses high frustration or dissatisfaction
   - You cannot resolve the issue after multiple attempts

RESPONSE FORMAT:
- Provide clear, concise answers
- Break down complex information into steps
- Offer additional help and next steps
- Maintain a friendly, professional tone

Remember: Customer satisfaction is the top priority."""
    
    def __init__(
        self,
        use_registry: Optional[bool] = None,
        session_store: Optional[SessionStore] = None
    ):
        """
        Initialize the agent with all tools.
        
        Args:
            use_registry: Whether to use registry mode (None = auto-detect from settings)
            session_store: Session store instance (None = create default)
        """
        self.tools = {}
        self.initialized = False
        
        # Determine initialization mode
        if use_registry is None:
            registry_mode = getattr(settings, 'agent_tool_registry_mode', 'legacy')
            self.use_registry = (registry_mode == 'registry')
        else:
            self.use_registry = use_registry
        
        # Initialize session store
        self.session_store = session_store
        if self.session_store is None:
            self.session_store = self._create_default_session_store()
        
        # Retry configuration for tool calls
        self.retry_config = RetryConfig(
            max_attempts=getattr(settings, 'agent_max_retries', 3),
            wait_multiplier=1.0,
            wait_min=1.0,
            wait_max=10.0,
            retry_exceptions=(Exception,)
        )
        
        logger.info(
            f"Agent initialization mode: {'registry' if self.use_registry else 'legacy'}, "
            f"session store: {type(self.session_store).__name__}"
        )
        
        # Initialize on creation (legacy mode only)
        if not self.use_registry:
            self._initialize_legacy()
    
    def _create_default_session_store(self) -> SessionStore:
        """
        Create default session store based on configuration.
        
        Returns:
            SessionStore instance
        """
        use_shared = getattr(settings, 'use_shared_context', False)
        
        if use_shared:
            # Try to use Redis store
            try:
                from ..session import RedisSessionStore, REDIS_AVAILABLE
                
                if REDIS_AVAILABLE:
                    redis_url = getattr(settings, 'redis_url', 'redis://localhost:6379/0')
                    store = RedisSessionStore(
                        redis_url=redis_url,
                        key_prefix="agent:session:",
                        default_ttl=getattr(settings, 'session_timeout_minutes', 30) * 60
                    )
                    logger.info("Using RedisSessionStore for shared context")
                    return store
                else:
                    logger.warning("Redis not available, falling back to InMemorySessionStore")
            except Exception as e:
                logger.error(f"Failed to create RedisSessionStore: {e}, falling back to InMemorySessionStore")
        
        # Use in-memory store
        store = InMemorySessionStore(
            max_sessions=getattr(settings, 'session_max_sessions', 10000),
            default_ttl=getattr(settings, 'session_timeout_minutes', 30) * 60
        )
        logger.info("Using InMemorySessionStore")
        return store
    
    async def initialize_async(self) -> None:
        """
        Initialize agent asynchronously (registry mode).
        Must be called explicitly when using registry mode.
        """
        if not self.use_registry:
            logger.warning("initialize_async called in legacy mode - tools already initialized")
            return
        
        try:
            logger.info("Initializing agent in registry mode...")
            await self._initialize_registry()
            self.initialized = True
            logger.info(f"✓ Agent initialized with {len(self.tools)} tools (registry mode)")
        except Exception as e:
            logger.error(f"Failed to initialize agent in registry mode: {e}", exc_info=True)
            raise
    
    def _initialize_legacy(self) -> None:
        """Initialize all tools using legacy method."""
        try:
            logger.info("Initializing agent tools (legacy mode)...")
            
            from ..tools import RAGTool, MemoryTool, AttachmentTool, EscalationTool
            
            if tool_settings.enable_rag_tool:
                self.tools['rag'] = RAGTool()
                logger.info("✓ RAG tool initialized")
            
            if tool_settings.enable_memory_tool:
                self.tools['memory'] = MemoryTool()
                logger.info("✓ Memory tool initialized")
            
            if tool_settings.enable_attachment_tool:
                self.tools['attachment'] = AttachmentTool()
                logger.info("✓ Attachment tool initialized")
            
            if tool_settings.enable_escalation_tool:
                self.tools['escalation'] = EscalationTool()
                logger.info("✓ Escalation tool initialized")
            
            self.initialized = True
            logger.info(f"Agent initialized with {len(self.tools)} tools (legacy mode)")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent (legacy mode): {e}", exc_info=True)
            raise
    
    async def _initialize_registry(self) -> None:
        """Initialize all tools using registry."""
        try:
            from ..tools.registry import ToolRegistry, ToolDependencies
            
            dependencies = ToolDependencies(
                settings=settings,
                tool_settings=tool_settings
            )
            
            self.tools = await ToolRegistry.create_and_initialize_tools(
                dependencies=dependencies,
                enabled_only=True,
                concurrent_init=True
            )
            
            if not self.tools:
                logger.warning("No tools were created by registry")
            
        except Exception as e:
            logger.error(f"Failed to initialize tools via registry: {e}", exc_info=True)
            raise
    
    async def get_or_create_session(
        self,
        session_id: str,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> SessionData:
        """
        Get or create session data using SessionStore.
        
        Args:
            session_id: Session identifier
            request_id: Request correlation ID
            user_id: User identifier
            
        Returns:
            SessionData instance
        """
        # Try to get existing session
        session_data = await self.session_store.get(session_id)
        
        if session_data:
            # Update request_id
            session_data.request_id = request_id
            logger.debug(
                f"Retrieved existing session {session_id}",
                extra={"session_id": session_id, "request_id": request_id}
            )
        else:
            # Create new session
            session_data = SessionData(
                session_id=session_id,
                user_id=user_id,
                thread_id=str(uuid.uuid4()),
                request_id=request_id,
                created_at=datetime.utcnow()
            )
            
            await self.session_store.set(session_id, session_data)
            
            logger.info(
                f"Created new session {session_id}",
                extra={"session_id": session_id, "request_id": request_id}
            )
        
        return session_data
    
    async def load_session_context(
        self,
        session_id: str,
        request_id: Optional[str] = None
    ) -> str:
        """Load conversation context from memory with telemetry."""
        try:
            memory_tool = self.tools.get('memory')
            if not memory_tool:
                logger.warning("Memory tool not available")
                return ""
            
            async with tool_call_context(
                tool_name='memory',
                operation='load_context',
                request_id=request_id,
                session_id=session_id
            ):
                summary = await memory_tool.summarize_session(session_id)
                memories = await memory_tool.retrieve_memories(
                    session_id=session_id,
                    content_type="context",
                    limit=5
                )
                
                if memories:
                    recent_context = "\nRecent conversation points:\n"
                    for memory in memories[:3]:
                        recent_context += f"- {memory['content']}\n"
                    summary += recent_context
                
                return summary
            
        except Exception as e:
            logger.error(
                f"Error loading session context: {e}",
                extra={"session_id": session_id, "request_id": request_id}
            )
            return ""
    
    async def search_knowledge_base(
        self,
        query: str,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search the knowledge base using RAG tool with telemetry."""
        try:
            rag_tool = self.tools.get('rag')
            if not rag_tool:
                logger.warning("RAG tool not available")
                return []
            
            result = await call_tool_with_wrapper(
                tool=rag_tool,
                method_name='search',
                request_id=request_id,
                session_id=session_id,
                retry_config=self.retry_config,
                timeout=30.0,
                query=query,
                k=k,
                threshold=0.7
            )
            
            if isinstance(result, ToolResult):
                if result.success:
                    return result.data.get("sources", [])
                else:
                    logger.error(
                        f"RAG search failed: {result.error}",
                        extra={"request_id": request_id, "session_id": session_id}
                    )
                    return []
            else:
                return result.get("sources", [])
            
        except Exception as e:
            logger.error(
                f"RAG search error: {e}",
                extra={"request_id": request_id, "session_id": session_id}
            )
            return []
    
    async def process_attachments(
        self,
        attachments: List[Dict[str, Any]],
        request_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """Process uploaded attachments with telemetry."""
        if not attachments:
            return ""
        
        attachment_tool = self.tools.get('attachment')
        rag_tool = self.tools.get('rag')
        
        if not attachment_tool:
            logger.warning("Attachment tool not available")
            return ""
        
        processed_content = "\n📎 Attached Documents:\n"
        
        for attachment in attachments:
            try:
                result = await call_tool_with_wrapper(
                    tool=attachment_tool,
                    method_name='process_attachment',
                    request_id=request_id,
                    session_id=session_id,
                    timeout=60.0,
                    file_path=attachment.get("path"),
                    filename=attachment.get("filename"),
                    chunk_for_rag=True
                )
                
                if isinstance(result, ToolResult):
                    result = result.data
                
                if result.get("success"):
                    processed_content += f"\n[{result['filename']}]:\n"
                    processed_content += f"{result.get('preview', '')}\n"
                    
                    if rag_tool and "chunks" in result:
                        rag_tool.add_documents(
                            documents=result["chunks"],
                            metadatas=[
                                {
                                    "source": result['filename'],
                                    "type": "user_upload",
                                    "session_id": session_id,
                                    "request_id": request_id
                                }
                                for _ in result["chunks"]
                            ]
                        )
                        logger.info(
                            f"Indexed {len(result['chunks'])} chunks from {result['filename']}",
                            extra={"request_id": request_id, "session_id": session_id}
                        )
                
            except Exception as e:
                logger.error(
                    f"Error processing attachment: {e}",
                    extra={"request_id": request_id, "session_id": session_id}
                )
                processed_content += f"\n[Error processing {attachment.get('filename', 'file')}]\n"
        
        return processed_content
    
    async def check_escalation(
        self,
        message: str,
        session_data: SessionData,
        message_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """Check if escalation is needed with telemetry."""
        try:
            escalation_tool = self.tools.get('escalation')
            if not escalation_tool:
                logger.warning("Escalation tool not available")
                return {"escalate": False, "confidence": 0.0}
            
            result = await call_tool_with_wrapper(
                tool=escalation_tool,
                method_name='should_escalate',
                request_id=session_data.request_id,
                session_id=session_data.session_id,
                timeout=10.0,
                message=message,
                message_history=message_history,
                metadata={
                    "session_id": session_data.session_id,
                    "message_count": session_data.message_count,
                    "already_escalated": session_data.escalated
                }
            )
            
            if isinstance(result, ToolResult):
                result = result.data
            
            if result.get("escalate") and not session_data.escalated:
                result["ticket"] = escalation_tool.create_escalation_ticket(
                    session_id=session_data.session_id,
                    escalation_result=result,
                    user_info={"user_id": session_data.user_id}
                )
                
                # Update session with escalation flag
                await self.session_store.update(
                    session_data.session_id,
                    {"escalated": True}
                )
                session_data.escalated = True
                
                logger.info(
                    f"Escalation triggered for session {session_data.session_id}",
                    extra={"session_id": session_data.session_id, "request_id": session_data.request_id}
                )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Escalation check error: {e}",
                extra={"session_id": session_data.session_id, "request_id": session_data.request_id}
            )
            return {"escalate": False, "confidence": 0.0}
    
    async def store_conversation_memory(
        self,
        session_id: str,
        user_message: str,
        agent_response: str,
        important_facts: List[str] = None,
        request_id: Optional[str] = None
    ) -> None:
        """Store important information in memory with telemetry."""
        try:
            memory_tool = self.tools.get('memory')
            if not memory_tool:
                logger.warning("Memory tool not available")
                return
            
            async with tool_call_context(
                tool_name='memory',
                operation='store_conversation',
                request_id=request_id,
                session_id=session_id
            ):
                await memory_tool.store_memory(
                    session_id=session_id,
                    content=f"User: {user_message[:200]}",
                    content_type="context",
                    importance=0.5
                )
                
                if len(agent_response) > 100:
                    await memory_tool.store_memory(
                        session_id=session_id,
                        content=f"Agent: {agent_response[:200]}",
                        content_type="context",
                        importance=0.4
                    )
                
                if important_facts:
                    for fact in important_facts:
                        await memory_tool.store_memory(
                            session_id=session_id,
                            content=fact,
                            content_type="fact",
                            importance=0.8
                        )
            
        except Exception as e:
            logger.error(
                f"Error storing memory: {e}",
                extra={"session_id": session_id, "request_id": request_id}
            )
    
    def extract_important_facts(
        self,
        message: str,
        response: str
    ) -> List[str]:
        """Extract important facts from conversation."""
        facts = []
        
        import re
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, message)
        for email in emails:
            facts.append(f"User email: {email}")
        
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phones = re.findall(phone_pattern, message)
        for phone in phones:
            facts.append(f"User phone: {phone}")
        
        order_pattern = r'\b(?:order|ticket|reference|confirmation)\s*#?\s*([A-Z0-9-]+)\b'
        orders = re.findall(order_pattern, message, re.IGNORECASE)
        for order in orders:
            facts.append(f"Reference number: {order}")
        
        return facts
    
    async def generate_response(
        self,
        message: str,
        context: str,
        sources: List[Dict],
        escalation: Dict[str, Any]
    ) -> str:
        """Generate agent response based on context and tools."""
        response_parts = []
        
        if context == "No previous context available for this session.":
            response_parts.append("Hello! I'm here to help you today.")
        
        if sources:
            response_parts.append("Based on our information:")
            for i, source in enumerate(sources[:2], 1):
                response_parts.append(f"{i}. {source['content'][:200]}...")
        
        if escalation.get("escalate"):
            response_parts.append(
                "\nI understand this is important to you. "
                "I'm connecting you with a human support specialist who can better assist you."
            )
            if escalation.get("ticket"):
                response_parts.append(
                    f"Your ticket number is: {escalation['ticket']['ticket_id']}"
                )
        
        if not response_parts:
            response_parts.append(
                "I'm here to help! Could you please provide more details about your inquiry?"
            )
        
        return "\n\n".join(response_parts)
    
    async def process_message(
        self,
        session_id: str,
        message: str,
        attachments: Optional[List[Dict[str, Any]]] = None,
        user_id: Optional[str] = None,
        message_history: Optional[List[Dict]] = None,
        request_id: Optional[str] = None
    ) -> AgentResponse:
        """
        Process a user message and generate response.
        
        Args:
            session_id: Session identifier
            message: User message
            attachments: Optional file attachments
            user_id: Optional user identifier
            message_history: Previous messages
            request_id: Request correlation ID
            
        Returns:
            AgentResponse with generated response and metadata
        """
        start_time = datetime.utcnow()
        
        if not request_id:
            request_id = str(uuid.uuid4())
        
        try:
            # Get or create session using SessionStore
            session_data = await self.get_or_create_session(session_id, request_id, user_id)
            
            # Atomically increment message count
            new_count = await self.session_store.increment_counter(
                session_id,
                'message_count',
                delta=1
            )
            session_data.message_count = new_count
            
            logger.info(
                f"Processing message for session {session_id}",
                extra={
                    "session_id": session_id,
                    "request_id": request_id,
                    "message_count": session_data.message_count
                }
            )
            
            # Load session context
            session_context = await self.load_session_context(session_id, request_id)
            
            # Process attachments
            attachment_context = ""
            if attachments:
                attachment_context = await self.process_attachments(
                    attachments,
                    request_id,
                    session_id
                )
            
            # Search knowledge base
            sources = await self.search_knowledge_base(
                message,
                request_id,
                session_id
            )
            
            # Check escalation
            escalation = await self.check_escalation(message, session_data, message_history)
            
            # Generate response
            response_text = await self.generate_response(
                message=message,
                context=session_context,
                sources=sources,
                escalation=escalation
            )
            
            # Extract and store important facts
            facts = self.extract_important_facts(message, response_text)
            await self.store_conversation_memory(
                session_id=session_id,
                user_message=message,
                agent_response=response_text,
                important_facts=facts,
                request_id=request_id
            )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Build response
            response = AgentResponse(
                message=response_text,
                sources=sources[:3],
                requires_escalation=escalation.get("escalate", False),
                confidence=escalation.get("confidence", 0.95),
                tools_used=list(self.tools.keys()),
                processing_time=processing_time
            )
            
            # Add metadata
            session_stats = await self.session_store.get_stats()
            
            response.tool_metadata = {
                "session_id": session_id,
                "request_id": request_id,
                "message_count": session_data.message_count,
                "has_context": bool(session_context),
                "facts_extracted": len(facts),
                "initialization_mode": "registry" if self.use_registry else "legacy",
                "session_store": type(self.session_store).__name__,
                "session_stats": session_stats,
                "circuit_breaker_status": get_tool_metrics()
            }
            
            if escalation.get("ticket"):
                response.tool_metadata["ticket_id"] = escalation["ticket"]["ticket_id"]
            
            logger.info(
                f"Processed message for session {session_id} in {processing_time:.2f}s",
                extra={
                    "session_id": session_id,
                    "request_id": request_id,
                    "processing_time": processing_time,
                    "escalated": response.requires_escalation
                }
            )
            
            return response
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.error(
                f"Error processing message: {e}",
                extra={
                    "session_id": session_id,
                    "request_id": request_id,
                    "processing_time": processing_time
                },
                exc_info=True
            )
            
            return AgentResponse(
                message="I apologize, but I encountered an error processing your request. "
                        "Please try again or contact support directly.",
                requires_escalation=True,
                confidence=0.0,
                processing_time=processing_time
            )
    
    async def stream_response(
        self,
        session_id: str,
        message: str,
        attachments: Optional[List[Dict[str, Any]]] = None,
        request_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response for real-time interaction with telemetry."""
        if not request_id:
            request_id = str(uuid.uuid4())
        
        try:
            yield {
                "type": "start",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            yield {
                "type": "status",
                "message": "Loading conversation context...",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            session_data = await self.get_or_create_session(session_id, request_id)
            session_context = await self.load_session_context(session_id, request_id)
            
            if attachments:
                yield {
                    "type": "status",
                    "message": "Processing attachments...",
                    "timestamp": datetime.utcnow().isoformat()
                }
                attachment_context = await self.process_attachments(
                    attachments,
                    request_id,
                    session_id
                )
            
            yield {
                "type": "status",
                "message": "Searching knowledge base...",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            sources = await self.search_knowledge_base(message, request_id, session_id)
            
            if sources:
                yield {
                    "type": "sources",
                    "sources": sources[:3],
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            escalation = await self.check_escalation(message, session_data)
            
            if escalation.get("escalate"):
                yield {
                    "type": "escalation",
                    "required": True,
                    "reason": escalation.get("reasons", []),
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            response_text = await self.generate_response(
                message=message,
                context=session_context,
                sources=sources,
                escalation=escalation
            )
            
            words = response_text.split()
            chunk_size = 5
            
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i+chunk_size])
                yield {
                    "type": "text",
                    "content": chunk + ' ',
                    "timestamp": datetime.utcnow().isoformat()
                }
                await asyncio.sleep(0.05)
            
            await self.store_conversation_memory(
                session_id=session_id,
                user_message=message,
                agent_response=response_text,
                request_id=request_id
            )
            
            yield {
                "type": "complete",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(
                f"Streaming error: {e}",
                extra={"session_id": session_id, "request_id": request_id}
            )
            yield {
                "type": "error",
                "error": str(e),
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def cleanup_session(self, session_id: str) -> None:
        """Clean up session resources."""
        try:
            # Delete from session store
            await self.session_store.delete(session_id)
            
            # Clean up old memories
            memory_tool = self.tools.get('memory')
            if memory_tool:
                await memory_tool.cleanup_old_memories(days=30)
            
            logger.info(f"Cleaned up session: {session_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up session: {e}")
    
    async def cleanup(self) -> None:
        """Clean up all agent resources."""
        logger.info("Cleaning up agent resources...")
        
        # Cleanup expired sessions
        try:
            cleaned = await self.session_store.cleanup_expired()
            logger.info(f"Cleaned up {cleaned} expired sessions")
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")
        
        # Cleanup tools
        if self.use_registry:
            from ..tools.registry import ToolRegistry
            await ToolRegistry.cleanup_tools(self.tools, concurrent=True)
        else:
            for tool_name, tool in self.tools.items():
                if hasattr(tool, 'cleanup'):
                    try:
                        await tool.cleanup()
                        logger.info(f"Cleaned up {tool_name} tool")
                    except Exception as e:
                        logger.error(f"Error cleaning up {tool_name} tool: {e}")
        
        # Close session store connection
        if hasattr(self.session_store, 'close'):
            try:
                await self.session_store.close()
                logger.info("Closed session store connection")
            except Exception as e:
                logger.error(f"Error closing session store: {e}")
        
        logger.info("Agent cleanup complete")
```

---

*Due to character limits, I'll continue with the remaining files in the next message.*

Would you like me to continue with:
- File 6: `backend/app/config.py` (UPDATE - Add session configuration)
- File 7: `requirements.txt` (UPDATE - Add redis dependency)
- File 8: `tests/test_session_store.py` (NEW - Comprehensive tests)
- Complete integration instructions and validation checklist
