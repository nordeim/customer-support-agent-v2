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
