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
