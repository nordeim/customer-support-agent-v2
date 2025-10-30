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
