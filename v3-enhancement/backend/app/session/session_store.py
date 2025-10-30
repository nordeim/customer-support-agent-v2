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
