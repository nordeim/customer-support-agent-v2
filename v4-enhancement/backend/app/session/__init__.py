"""
Session management package.
Provides session storage abstractions for agent context persistence.

Version: 2.0.0 (Enhanced with validation, encryption, and locking)
"""
from .session_store import SessionStore
from .validators import SessionData, SessionMetrics, SessionFilter
from .in_memory_session_store import InMemorySessionStore

# Conditionally import Redis store
try:
    from .redis_session_store import RedisSessionStore, REDIS_AVAILABLE
except ImportError:
    RedisSessionStore = None
    REDIS_AVAILABLE = False

# Conditionally import distributed lock
try:
    from .distributed_lock import (
        DistributedLock,
        DistributedLockManager,
        LockAcquisitionError,
        LockReleaseError
    )
    DISTRIBUTED_LOCK_AVAILABLE = True
except ImportError:
    DistributedLock = None
    DistributedLockManager = None
    LockAcquisitionError = None
    LockReleaseError = None
    DISTRIBUTED_LOCK_AVAILABLE = False


def create_session_store(
    store_type: str = "in_memory",
    **kwargs
) -> SessionStore:
    """
    Factory function to create session store.
    
    Args:
        store_type: Type of store ('in_memory' or 'redis')
        **kwargs: Store-specific configuration
        
    Returns:
        SessionStore instance
        
    Examples:
        # In-memory store
        store = create_session_store('in_memory', max_sessions=10000)
        
        # Redis store
        store = create_session_store(
            'redis',
            redis_url='redis://localhost:6379/0',
            enable_l1_cache=True,
            enable_encryption=True,
            encryption_key='your-key-here'
        )
    """
    if store_type == "in_memory":
        return InMemorySessionStore(**kwargs)
    
    elif store_type == "redis":
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis support not available. "
                "Install with: pip install redis[asyncio]"
            )
        return RedisSessionStore(**kwargs)
    
    else:
        raise ValueError(f"Unknown store type: {store_type}")


__all__ = [
    # Core
    'SessionStore',
    'SessionData',
    'SessionMetrics',
    'SessionFilter',
    
    # Implementations
    'InMemorySessionStore',
    'RedisSessionStore',
    
    # Distributed locking
    'DistributedLock',
    'DistributedLockManager',
    'LockAcquisitionError',
    'LockReleaseError',
    
    # Factory
    'create_session_store',
    
    # Availability flags
    'REDIS_AVAILABLE',
    'DISTRIBUTED_LOCK_AVAILABLE'
]
