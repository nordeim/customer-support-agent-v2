"""
Memory management tool for conversation context persistence.
Uses database for storing and retrieving conversation memories.

Version: 2.0.0 (Connection pooling, duplicate handling, field naming fixed)
"""
import logging
import json
import uuid
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, desc, and_, or_, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool
from sqlalchemy.exc import IntegrityError

from ..config import settings
from ..models.memory import Memory
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
    
    Version 2.0.0:
    - FIXED: Connection pooling for production
    - FIXED: Unique constraint handling for duplicates
    - FIXED: Field naming (metadata, not tool_metadata)
    - Enhanced error handling
    """
    
    def __init__(self):
        """Initialize memory tool with database connection."""
        super().__init__(
            name="memory_management",
            description="Store and retrieve conversation memory and context",
            version="2.0.0"
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
    # Core Memory Methods (Async)
    # ===========================
    
    async def store_memory_async(
        self,
        session_id: str,
        content: str,
        content_type: str = "context",
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5
    ) -> Dict[str, Any]:
        """Store a memory entry for a session (async)."""
        if content_type not in MEMORY_TYPE_PRIORITY:
            return {
                "success": False,
                "error": f"Invalid content_type. Must be one of: {list(MEMORY_TYPE_PRIORITY.keys())}"
            }
        
        if not (0.0 <= importance <= 1.0):
            importance = max(0.0, min(1.0, importance))
        
        try:
            # Run database operation in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._store_memory_sync,
                session_id,
                content,
                content_type,
                metadata,
                importance
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
                "error": str(e)
            }
    
    async def retrieve_memories_async(
        self,
        session_id: str,
        content_type: Optional[str] = None,
        limit: int = DEFAULT_MEMORY_LIMIT,
        time_window_hours: Optional[int] = None,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Retrieve memories for a session (async)."""
        try:
            # Run database operation in thread pool
            memories = await asyncio.get_event_loop().run_in_executor(
                None,
                self._retrieve_memories_sync,
                session_id,
                content_type,
                limit,
                time_window_hours,
                min_importance
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
        """Generate a text summary of session memories (async)."""
        try:
            # Retrieve memories grouped by type
            memory_groups = {}
            
            for content_type in MEMORY_TYPE_PRIORITY.keys():
                memories = await self.retrieve_memories_async(
                    session_id=session_id,
                    content_type=content_type,
                    limit=max_items_per_type,
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
    # Private Helper Methods (Sync)
    # ===========================
    
    def _init_database(self) -> None:
        """
        Initialize database engine with proper connection pooling.
        
        Version 2.0.0: FIXED - Added connection pool configuration.
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
                # PostgreSQL or other databases - FIXED: Added pool configuration
                pool_config = {
                    "pool_size": 10,  # Base pool size
                    "max_overflow": 20,  # Additional connections
                    "pool_timeout": 30,  # Seconds to wait for connection
                    "pool_recycle": 3600,  # Recycle connections after 1 hour
                    "pool_pre_ping": True  # Test connections before use
                }
                poolclass = QueuePool
            
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
                expire_on_commit=False  # Performance optimization
            )
            
            pool_size = pool_config.get('pool_size', 'N/A')
            logger.info(f"Memory database initialized (pool_size={pool_size})")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}", exc_info=True)
            raise
    
    def _get_db_session(self) -> Session:
        """Get database session."""
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self.SessionLocal()
    
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
        
        Version 2.0.0: FIXED - Handle unique constraint violations.
        """
        db = self._get_db_session()
        try:
            # Try to create new memory
            memory = Memory(
                id=str(uuid.uuid4()),
                session_id=session_id,
                content_type=content_type,
                content=content,
                metadata=metadata or {},  # FIXED: correct field name
                importance=importance
            )
            
            db.add(memory)
            
            try:
                db.commit()
                
                return {
                    "success": True,
                    "memory_id": memory.id,
                    "action": "created",
                    "message": "Memory stored successfully"
                }
                
            except IntegrityError:
                # FIXED: Duplicate detected, update existing
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
                    
                    db.commit()
                    
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
            
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()
    
    def _retrieve_memories_sync(
        self,
        session_id: str,
        content_type: Optional[str],
        limit: int,
        time_window_hours: Optional[int],
        min_importance: float
    ) -> List[Dict[str, Any]]:
        """Retrieve memories (sync implementation for thread pool)."""
        db = self._get_db_session()
        try:
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
            
            db.commit()
            
            # Format results - FIXED: use correct field name
            results = []
            for memory in memories:
                results.append({
                    "id": memory.id,
                    "content_type": memory.content_type,
                    "content": memory.content,
                    "metadata": memory.metadata,  # FIXED: correct field name
                    "importance": memory.importance,
                    "created_at": memory.created_at.isoformat(),
                    "access_count": memory.access_count
                })
            
            return results
            
        finally:
            db.close()
    
    def _cleanup_old_memories_sync(
        self,
        days: int,
        max_per_session: int
    ) -> Dict[str, Any]:
        """Cleanup old memories (sync implementation for thread pool)."""
        db = self._get_db_session()
        try:
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
            
            db.commit()
            
            total_deleted = deleted_old + deleted_excess
            
            return {
                "success": True,
                "deleted_old": deleted_old,
                "deleted_excess": deleted_excess,
                "total_deleted": total_deleted
            }
            
        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()


__all__ = ['MemoryTool']
