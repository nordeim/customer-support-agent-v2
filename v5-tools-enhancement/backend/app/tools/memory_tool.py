"""
Memory management tool for conversation context persistence.
Uses database for storing and retrieving conversation memories.

Version: 3.0.0 (Enhanced with validation, context managers, and security)

Changes:
- Added Pydantic input validation
- Added database session context managers with timeouts
- Added query timeout configuration
- Enhanced error handling and logging
- Improved connection health checks
"""
import logging
import json
import uuid
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from contextlib import contextmanager

from sqlalchemy import create_engine, desc, and_, or_, func, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool
from sqlalchemy.exc import IntegrityError, DBAPIError, DisconnectionError
from pydantic import ValidationError

from ..config import settings
from ..models.memory import Memory
from ..schemas.tool_requests import (
    MemoryStoreRequest,
    MemoryRetrieveRequest,
    MemorySummarizeRequest
)
from .base_tool import BaseTool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)

# Memory type priorities for retrieval
MEMORY_TYPE_PRIORITY = {
    "user_info": 1.0,
    "preference": 0.9,
    "fact": 0.8,
    "context": 0.7
}

# Default limits
DEFAULT_MEMORY_LIMIT = 10
DEFAULT_TIME_WINDOW_HOURS = 24


class MemoryTool(BaseTool):
    """
    Memory management tool for storing and retrieving conversation context.
    
    Version 3.0.0:
    - ADDED: Pydantic input validation
    - ADDED: Database session context managers with timeouts
    - FIXED: Connection pooling for production
    - FIXED: Unique constraint handling for duplicates
    - FIXED: Field naming (metadata, not tool_metadata)
    - Enhanced error handling and security
    """
    
    def __init__(self):
        """Initialize memory tool with database connection."""
        super().__init__(
            name="memory_management",
            description="Store and retrieve conversation memory and context",
            version="3.0.0"
        )
        
        # Resources initialized in async initialize()
        self.engine = None
        self.SessionLocal = None
    
    async def initialize(self) -> None:
        """Initialize memory tool resources (async-safe)."""
        try:
            logger.info(f"Initializing Memory tool '{self.name}'...")
            
            # Initialize database engine (I/O-bound, run in thread pool)
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._init_database
            )
            
            self.initialized = True
            logger.info(f"✓ Memory tool '{self.name}' initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Memory tool: {e}", exc_info=True)
            raise
    
    async def cleanup(self) -> None:
        """Cleanup memory tool resources."""
        try:
            logger.info(f"Cleaning up Memory tool '{self.name}'...")
            
            # Dispose of database engine
            if self.engine:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.engine.dispose
                )
                self.engine = None
                self.SessionLocal = None
            
            self.initialized = False
            logger.info(f"✓ Memory tool '{self.name}' cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during Memory tool cleanup: {e}")
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute memory operations (async-first)."""
        action = kwargs.get("action", "retrieve")
        session_id = kwargs.get("session_id")
        
        if not session_id:
            return ToolResult.error_result(
                error="session_id is required",
                metadata={"tool": self.name}
            )
        
        try:
            if action == "store":
                content = kwargs.get("content")
                if not content:
                    return ToolResult.error_result(
                        error="content is required for store action",
                        metadata={"tool": self.name, "action": action}
                    )
                
                result = await self.store_memory_async(
                    session_id=session_id,
                    content=content,
                    content_type=kwargs.get("content_type", "context"),
                    metadata=kwargs.get("metadata"),
                    importance=kwargs.get("importance", 0.5)
                )
                
                return ToolResult.success_result(
                    data=result,
                    metadata={
                        "tool": self.name,
                        "action": action,
                        "session_id": session_id
                    }
                )
            
            elif action == "retrieve":
                memories = await self.retrieve_memories_async(
                    session_id=session_id,
                    content_type=kwargs.get("content_type"),
                    limit=kwargs.get("limit", DEFAULT_MEMORY_LIMIT),
                    time_window_hours=kwargs.get("time_window_hours"),
                    min_importance=kwargs.get("min_importance", 0.0)
                )
                
                return ToolResult.success_result(
                    data={
                        "memories": memories,
                        "count": len(memories)
                    },
                    metadata={
                        "tool": self.name,
                        "action": action,
                        "session_id": session_id
                    }
                )
            
            elif action == "summarize":
                summary = await self.summarize_session_async(
                    session_id=session_id,
                    max_items_per_type=kwargs.get("max_items_per_type", 3)
                )
                
                return ToolResult.success_result(
                    data={"summary": summary},
                    metadata={
                        "tool": self.name,
                        "action": action,
                        "session_id": session_id,
                        "summary_length": len(summary)
                    }
                )
            
            else:
                return ToolResult.error_result(
                    error=f"Unknown action: {action}. Valid: store, retrieve, summarize",
                    metadata={"tool": self.name}
                )
                
        except Exception as e:
            logger.error(f"Memory execute error: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={"tool": self.name, "action": action, "session_id": session_id}
            )
    
    # ===========================
    # Core Memory Methods (Async) - ENHANCED WITH VALIDATION
    # ===========================
    
    async def store_memory_async(
        self,
        session_id: str,
        content: str,
        content_type: str = "context",
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5
    ) -> Dict[str, Any]:
        """
        Store a memory entry for a session (async).
        
        Version 3.0.0: Added Pydantic validation.
        """
        # ADDED: Input validation
        try:
            validated_request = MemoryStoreRequest(
                session_id=session_id,
                content=content,
                content_type=content_type,
                metadata=metadata,
                importance=importance
            )
        except ValidationError as e:
            logger.error(f"Memory store validation failed: {e}")
            return {
                "success": False,
                "error": f"Input validation failed: {e}",
                "validation_errors": e.errors()
            }
        
        try:
            # Run database operation in thread pool with validated data
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._store_memory_sync,
                validated_request.session_id,
                validated_request.content,
                validated_request.content_type,
                validated_request.metadata,
                validated_request.importance
            )
            
            logger.info(
                f"Stored memory for session {session_id}: "
                f"type={content_type}, importance={importance}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def retrieve_memories_async(
        self,
        session_id: str,
        content_type: Optional[str] = None,
        limit: int = DEFAULT_MEMORY_LIMIT,
        time_window_hours: Optional[int] = None,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories for a session (async).
        
        Version 3.0.0: Added Pydantic validation.
        """
        # ADDED: Input validation
        try:
            validated_request = MemoryRetrieveRequest(
                session_id=session_id,
                content_type=content_type,
                limit=limit,
                time_window_hours=time_window_hours,
                min_importance=min_importance
            )
        except ValidationError as e:
            logger.error(f"Memory retrieve validation failed: {e}")
            return []
        
        try:
            # Run database operation in thread pool with validated data
            memories = await asyncio.get_event_loop().run_in_executor(
                None,
                self._retrieve_memories_sync,
                validated_request.session_id,
                validated_request.content_type,
                validated_request.limit,
                validated_request.time_window_hours,
                validated_request.min_importance
            )
            
            logger.debug(f"Retrieved {len(memories)} memories for session {session_id}")
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}", exc_info=True)
            return []
    
    async def summarize_session_async(
        self,
        session_id: str,
        max_items_per_type: int = 3
    ) -> str:
        """
        Generate a text summary of session memories (async).
        
        Version 3.0.0: Added Pydantic validation.
        """
        # ADDED: Input validation
        try:
            validated_request = MemorySummarizeRequest(
                session_id=session_id,
                max_items_per_type=max_items_per_type
            )
        except ValidationError as e:
            logger.error(f"Memory summarize validation failed: {e}")
            return "Error: Invalid request parameters."
        
        try:
            # Retrieve memories grouped by type
            memory_groups = {}
            
            for content_type in MEMORY_TYPE_PRIORITY.keys():
                memories = await self.retrieve_memories_async(
                    session_id=validated_request.session_id,
                    content_type=content_type,
                    limit=validated_request.max_items_per_type,
                    min_importance=0.3
                )
                
                if memories:
                    memory_groups[content_type] = memories
            
            if not memory_groups:
                return "No previous context available for this session."
            
            # Build summary
            summary_parts = []
            
            if "user_info" in memory_groups:
                user_info = [m["content"] for m in memory_groups["user_info"]]
                summary_parts.append(f"User Information: {'; '.join(user_info)}")
            
            if "preference" in memory_groups:
                preferences = [m["content"] for m in memory_groups["preference"]]
                summary_parts.append(f"User Preferences: {'; '.join(preferences)}")
            
            if "fact" in memory_groups:
                facts = [m["content"] for m in memory_groups["fact"][:3]]
                summary_parts.append(f"Key Facts: {'; '.join(facts)}")
            
            if "context" in memory_groups:
                contexts = [m["content"] for m in memory_groups["context"][:5]]
                summary_parts.append(f"Recent Context: {'; '.join(contexts[:3])}")
            
            summary = "\n".join(summary_parts)
            
            logger.debug(f"Generated summary for session {session_id}: {len(summary)} chars")
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate session summary: {e}", exc_info=True)
            return "Error retrieving session context."
    
    async def cleanup_old_memories_async(
        self,
        days: int = 30,
        max_per_session: int = 100
    ) -> Dict[str, Any]:
        """Clean up old and low-importance memories (async)."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._cleanup_old_memories_sync,
                days,
                max_per_session
            )
            
            logger.info(f"Memory cleanup completed: {result['total_deleted']} memories deleted")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to cleanup memories: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    # ===========================
    # Database Session Context Manager (NEW)
    # ===========================
    
    @contextmanager
    def get_db_session_context(self, timeout: float = 30.0):
        """
        Context manager for database sessions with timeout and health checks.
        
        ADDED in Version 3.0.0 for safe session management.
        
        Args:
            timeout: Maximum session lifetime in seconds
            
        Yields:
            Database session
            
        Raises:
            RuntimeError: If session times out or database not initialized
            
        Example:
            with self.get_db_session_context(timeout=10.0) as db:
                memory = Memory(...)
                db.add(memory)
                # Commit happens automatically if no exception
        """
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        session = self.SessionLocal()
        start_time = time.time()
        
        try:
            # Validate connection with a simple query
            try:
                session.execute(text("SELECT 1"))
            except (DBAPIError, DisconnectionError) as e:
                logger.error(f"Database connection validation failed: {e}")
                session.invalidate()
                raise RuntimeError(f"Database connection unhealthy: {e}")
            
            # Yield session for operations
            yield session
            
            # Check if timeout exceeded
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(f"Session exceeded timeout: {elapsed:.2f}s > {timeout}s")
                session.rollback()
                raise RuntimeError(f"Session timeout exceeded: {elapsed:.2f}s")
            
            # Commit if no exceptions
            session.commit()
            
        except (DBAPIError, DisconnectionError) as e:
            logger.error(f"Database error during session: {e}")
            session.rollback()
            # Test if connection is still alive
            try:
                session.execute(text("SELECT 1"))
            except Exception:
                # Connection is dead, invalidate it
                session.invalidate()
            raise
            
        except Exception as e:
            logger.error(f"Session error: {e}", exc_info=True)
            session.rollback()
            raise
            
        finally:
            # GUARANTEED cleanup
            session.close()
            
            # Log slow sessions
            elapsed = time.time() - start_time
            if elapsed > 5.0:
                logger.warning(f"Slow database session: {elapsed:.2f}s")
    
    # ===========================
    # Private Helper Methods (Sync) - UPDATED WITH CONTEXT MANAGERS
    # ===========================
    
    def _init_database(self) -> None:
        """
        Initialize database engine with proper connection pooling.
        
        Version 3.0.0: Added query timeout configuration.
        """
        try:
            connect_args = {}
            poolclass = None
            pool_config = {}
            
            if "sqlite" in settings.database_url:
                # SQLite configuration
                connect_args = {
                    "check_same_thread": False,
                    "timeout": 20
                }
                poolclass = StaticPool  # SQLite doesn't benefit from pooling
            else:
                # PostgreSQL or other databases
                pool_config = {
                    "pool_size": 10,
                    "max_overflow": 20,
                    "pool_timeout": 30,
                    "pool_recycle": 3600,
                    "pool_pre_ping": True
                }
                poolclass = QueuePool
                
                # ADDED: Query timeout for PostgreSQL
                connect_args = {
                    "options": "-c statement_timeout=30000"  # 30 seconds
                }
            
            self.engine = create_engine(
                settings.database_url,
                connect_args=connect_args,
                poolclass=poolclass,
                echo=settings.database_echo,
                **pool_config
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
                expire_on_commit=False
            )
            
            pool_size = pool_config.get('pool_size', 'N/A')
            logger.info(f"Memory database initialized (pool_size={pool_size})")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}", exc_info=True)
            raise
    
    def _store_memory_sync(
        self,
        session_id: str,
        content: str,
        content_type: str,
        metadata: Optional[Dict[str, Any]],
        importance: float
    ) -> Dict[str, Any]:
        """
        Store memory (sync implementation for thread pool).
        
        Version 3.0.0: UPDATED to use context manager.
        """
        # UPDATED: Use context manager for safe session management
        with self.get_db_session_context(timeout=10.0) as db:
            try:
                # Try to create new memory
                memory = Memory(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    content_type=content_type,
                    content=content,
                    metadata=metadata or {},
                    importance=importance
                )
                
                db.add(memory)
                db.flush()  # Get any database errors early
                
                return {
                    "success": True,
                    "memory_id": memory.id,
                    "action": "created",
                    "message": "Memory stored successfully"
                }
                
            except IntegrityError:
                # Duplicate detected, update existing
                db.rollback()
                
                existing = db.query(Memory).filter(
                    and_(
                        Memory.session_id == session_id,
                        Memory.content_type == content_type,
                        Memory.content == content
                    )
                ).first()
                
                if existing:
                    # Update importance and access tracking
                    existing.importance = max(existing.importance, importance)
                    existing.last_accessed = datetime.utcnow()
                    existing.access_count += 1
                    
                    # Update metadata if provided
                    if metadata:
                        existing.metadata.update(metadata)
                    
                    db.flush()
                    
                    logger.debug(f"Updated existing memory: {existing.id}")
                    
                    return {
                        "success": True,
                        "memory_id": existing.id,
                        "action": "updated",
                        "message": "Memory updated successfully"
                    }
                else:
                    # Race condition: deleted between attempts
                    raise
    
    def _retrieve_memories_sync(
        self,
        session_id: str,
        content_type: Optional[str],
        limit: int,
        time_window_hours: Optional[int],
        min_importance: float
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories (sync implementation for thread pool).
        
        Version 3.0.0: UPDATED to use context manager.
        """
        # UPDATED: Use context manager
        with self.get_db_session_context(timeout=10.0) as db:
            query = db.query(Memory).filter(Memory.session_id == session_id)
            
            # Apply filters
            if content_type:
                query = query.filter(Memory.content_type == content_type)
            
            if time_window_hours:
                cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
                query = query.filter(Memory.created_at >= cutoff_time)
            
            if min_importance > 0:
                query = query.filter(Memory.importance >= min_importance)
            
            # Order by importance and recency
            query = query.order_by(
                desc(Memory.importance),
                desc(Memory.created_at)
            ).limit(limit)
            
            memories = query.all()
            
            # Update access times
            for memory in memories:
                memory.update_access()
            
            db.flush()
            
            # Format results
            results = []
            for memory in memories:
                results.append({
                    "id": memory.id,
                    "content_type": memory.content_type,
                    "content": memory.content,
                    "metadata": memory.metadata,
                    "importance": memory.importance,
                    "created_at": memory.created_at.isoformat(),
                    "access_count": memory.access_count
                })
            
            return results
    
    def _cleanup_old_memories_sync(
        self,
        days: int,
        max_per_session: int
    ) -> Dict[str, Any]:
        """
        Cleanup old memories (sync implementation for thread pool).
        
        Version 3.0.0: UPDATED to use context manager with longer timeout.
        """
        # UPDATED: Use context manager with longer timeout for cleanup
        with self.get_db_session_context(timeout=60.0) as db:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Delete old, low-importance, rarely accessed memories
            deleted_old = db.query(Memory).filter(
                and_(
                    Memory.last_accessed < cutoff_date,
                    Memory.importance < 0.3,
                    Memory.access_count < 3
                )
            ).delete(synchronize_session=False)
            
            # For each session, keep only the most recent/important memories
            sessions = db.query(Memory.session_id).distinct().all()
            deleted_excess = 0
            
            for (session_id,) in sessions:
                # Get memories ordered by importance and recency
                memories = db.query(Memory).filter(
                    Memory.session_id == session_id
                ).order_by(
                    desc(Memory.importance),
                    desc(Memory.created_at)
                ).offset(max_per_session).all()
                
                # Delete excess memories
                for memory in memories:
                    db.delete(memory)
                    deleted_excess += 1
            
            db.flush()
            
            total_deleted = deleted_old + deleted_excess
            
            return {
                "success": True,
                "deleted_old": deleted_old,
                "deleted_excess": deleted_excess,
                "total_deleted": total_deleted
            }


__all__ = ['MemoryTool']
