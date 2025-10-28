"""
Memory management tool for conversation context persistence.
Uses SQLite for storing and retrieving conversation memories.
"""
import logging
import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from contextlib import contextmanager

from sqlalchemy import create_engine, desc, and_, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from ..config import settings
from ..models.memory import Base, Memory
from .base_tool import BaseTool

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
    Provides persistent storage of important information across sessions.
    """
    
    def __init__(self):
        """Initialize memory tool with SQLite database."""
        super().__init__(
            name="memory_management",
            description="Store and retrieve conversation memory and context"
        )
    
    def _setup(self) -> None:
        """Setup SQLite database and create tables."""
        # Initialize database engine
        # Use StaticPool for SQLite to avoid threading issues
        self.engine = create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
            poolclass=StaticPool if "sqlite" in settings.database_url else None,
            echo=settings.database_echo
        )
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info(f"Memory tool initialized with database: {settings.database_url}")
    
    @contextmanager
    def get_db(self) -> Session:
        """
        Get database session with proper cleanup.
        
        Yields:
            Database session
        """
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    async def store_memory(
        self,
        session_id: str,
        content: str,
        content_type: str = "context",
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5
    ) -> Dict[str, Any]:
        """
        Store a memory entry for a session.
        
        Args:
            session_id: Session identifier
            content: Memory content to store
            content_type: Type of memory ('user_info', 'preference', 'context', 'fact')
            metadata: Optional metadata dictionary
            importance: Importance score (0.0 to 1.0)
            
        Returns:
            Status dictionary with memory ID
        """
        if content_type not in MEMORY_TYPE_PRIORITY:
            return {
                "success": False,
                "error": f"Invalid content_type. Must be one of: {list(MEMORY_TYPE_PRIORITY.keys())}"
            }
        
        if not (0.0 <= importance <= 1.0):
            importance = max(0.0, min(1.0, importance))
        
        try:
            with self.get_db() as db:
                # Check for duplicate memories (same content and type)
                existing = db.query(Memory).filter(
                    and_(
                        Memory.session_id == session_id,
                        Memory.content_type == content_type,
                        Memory.content == content
                    )
                ).first()
                
                if existing:
                    # Update importance and access time instead of creating duplicate
                    existing.importance = max(existing.importance, importance)
                    existing.last_accessed = datetime.utcnow()
                    existing.access_count += 1
                    db.commit()
                    
                    logger.debug(f"Updated existing memory: {existing.id}")
                    
                    return {
                        "success": True,
                        "memory_id": existing.id,
                        "action": "updated",
                        "message": "Memory updated successfully"
                    }
                
                # Create new memory
                memory = Memory(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    content_type=content_type,
                    content=content,
                    metadata=metadata or {},
                    importance=importance
                )
                
                db.add(memory)
                db.commit()
                
                logger.info(
                    f"Stored memory for session {session_id}: "
                    f"type={content_type}, importance={importance}"
                )
                
                return {
                    "success": True,
                    "memory_id": memory.id,
                    "action": "created",
                    "message": "Memory stored successfully"
                }
                
        except Exception as e:
            logger.error(f"Failed to store memory: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def retrieve_memories(
        self,
        session_id: str,
        content_type: Optional[str] = None,
        limit: int = DEFAULT_MEMORY_LIMIT,
        time_window_hours: Optional[int] = None,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories for a session.
        
        Args:
            session_id: Session identifier
            content_type: Filter by memory type (optional)
            limit: Maximum number of memories to retrieve
            time_window_hours: Only retrieve memories from last N hours (optional)
            min_importance: Minimum importance threshold
            
        Returns:
            List of memory dictionaries
        """
        try:
            with self.get_db() as db:
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
                    memory.last_accessed = datetime.utcnow()
                    memory.access_count += 1
                
                db.commit()
                
                # Format results
                results = []
                for memory in memories:
                    results.append({
                        "id": memory.id,
                        "content_type": memory.content_type,
                        "content": memory.content,
                        "metadata": memory.tool_metadata,
                        "importance": memory.importance,
                        "created_at": memory.created_at.isoformat(),
                        "access_count": memory.access_count
                    })
                
                logger.debug(
                    f"Retrieved {len(results)} memories for session {session_id}"
                )
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}", exc_info=True)
            return []
    
    async def summarize_session(
        self,
        session_id: str,
        max_items_per_type: int = 3
    ) -> str:
        """
        Generate a text summary of session memories.
        
        Args:
            session_id: Session identifier
            max_items_per_type: Maximum items per memory type
            
        Returns:
            Text summary of session context
        """
        try:
            # Retrieve memories grouped by type
            memory_groups = {}
            
            for content_type in MEMORY_TYPE_PRIORITY.keys():
                memories = await self.retrieve_memories(
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
    
    async def update_importance(
        self,
        memory_id: str,
        importance_delta: float
    ) -> Dict[str, Any]:
        """
        Update the importance score of a memory.
        
        Args:
            memory_id: Memory identifier
            importance_delta: Change in importance (-1.0 to 1.0)
            
        Returns:
            Status dictionary
        """
        try:
            with self.get_db() as db:
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                
                if not memory:
                    return {
                        "success": False,
                        "error": "Memory not found"
                    }
                
                # Update importance (keep within bounds)
                new_importance = max(0.0, min(1.0, memory.importance + importance_delta))
                memory.importance = new_importance
                memory.updated_at = datetime.utcnow()
                
                db.commit()
                
                logger.debug(f"Updated memory importance: {memory_id} -> {new_importance}")
                
                return {
                    "success": True,
                    "memory_id": memory_id,
                    "new_importance": new_importance
                }
                
        except Exception as e:
            logger.error(f"Failed to update memory importance: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def cleanup_old_memories(
        self,
        days: int = 30,
        max_per_session: int = 100
    ) -> Dict[str, Any]:
        """
        Clean up old and low-importance memories.
        
        Args:
            days: Delete memories older than N days with low importance
            max_per_session: Maximum memories to keep per session
            
        Returns:
            Cleanup statistics
        """
        try:
            with self.get_db() as db:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Delete old, low-importance, rarely accessed memories
                deleted_old = db.query(Memory).filter(
                    and_(
                        Memory.last_accessed < cutoff_date,
                        Memory.importance < 0.3,
                        Memory.access_count < 3
                    )
                ).delete()
                
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
                
                logger.info(
                    f"Memory cleanup completed: {total_deleted} memories deleted "
                    f"({deleted_old} old, {deleted_excess} excess)"
                )
                
                return {
                    "success": True,
                    "deleted_old": deleted_old,
                    "deleted_excess": deleted_excess,
                    "total_deleted": total_deleted
                }
                
        except Exception as e:
            logger.error(f"Failed to cleanup memories: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def execute(self, **kwargs) -> Any:
        """
        Execute memory operations.
        
        Accepts:
            action: Operation to perform ('store', 'retrieve', 'summarize')
            session_id: Session identifier (required)
            Other parameters based on action
            
        Returns:
            Operation results
        """
        action = kwargs.get("action", "retrieve")
        session_id = kwargs.get("session_id")
        
        if not session_id:
            return {
                "success": False,
                "error": "session_id is required"
            }
        
        if action == "store":
            content = kwargs.get("content")
            if not content:
                return {
                    "success": False,
                    "error": "content is required for store action"
                }
            
            return await self.store_memory(
                session_id=session_id,
                content=content,
                content_type=kwargs.get("content_type", "context"),
                metadata=kwargs.get("metadata"),
                importance=kwargs.get("importance", 0.5)
            )
        
        elif action == "retrieve":
            memories = await self.retrieve_memories(
                session_id=session_id,
                content_type=kwargs.get("content_type"),
                limit=kwargs.get("limit", DEFAULT_MEMORY_LIMIT),
                time_window_hours=kwargs.get("time_window_hours"),
                min_importance=kwargs.get("min_importance", 0.0)
            )
            
            return {
                "success": True,
                "memories": memories,
                "count": len(memories)
            }
        
        elif action == "summarize":
            summary = await self.summarize_session(
                session_id=session_id,
                max_items_per_type=kwargs.get("max_items_per_type", 3)
            )
            
            return {
                "success": True,
                "summary": summary
            }
        
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}. Valid actions: store, retrieve, summarize"
            }
