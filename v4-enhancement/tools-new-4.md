# 🚀 COMPLETE FILE IMPLEMENTATIONS (Final Files)

## File 10: `backend/app/tools/memory_tool.py` (Connection Pooling Fixed - Complete Replacement)

```python
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
```

---

## File 11: `backend/app/tools/escalation_tool.py` (Optimized - Complete Replacement)

```python
"""
Escalation tool for detecting when human intervention is needed.
Analyzes conversation context to determine escalation requirements.

Version: 2.0.0 (Optimized - removed unnecessary thread pool)
"""
import logging
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ..config import settings
from .base_tool import BaseTool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)

# Escalation trigger keywords and phrases
ESCALATION_KEYWORDS = {
    "urgent": 1.0,
    "emergency": 1.0,
    "complaint": 0.9,
    "angry": 0.9,
    "frustrated": 0.8,
    "disappointed": 0.8,
    "unacceptable": 0.8,
    "legal": 0.9,
    "lawsuit": 1.0,
    "lawyer": 0.9,
    "sue": 0.9,
    "refund": 0.7,
    "compensation": 0.7,
    "manager": 0.8,
    "supervisor": 0.8,
    "human": 0.7,
    "speak to someone": 0.8,
    "talk to a person": 0.8,
    "not helping": 0.7,
    "doesn't work": 0.6,
    "broken": 0.6,
    "critical": 0.9,
    "immediate": 0.8,
    "asap": 0.8,
    "right now": 0.8
}

# Sentiment thresholds
NEGATIVE_SENTIMENT_THRESHOLD = -0.5
ESCALATION_CONFIDENCE_THRESHOLD = 0.7


class EscalationTool(BaseTool):
    """
    Tool for detecting when a conversation should be escalated.
    
    Version 2.0.0:
    - FIXED: Removed unnecessary thread pool for _analyze_message
    - Optimized performance by running analysis directly
    - Enhanced error handling
    """
    
    def __init__(self):
        """Initialize escalation detection tool."""
        super().__init__(
            name="escalation_check",
            description="Determine if human intervention is needed",
            version="2.0.0"
        )
        
        # Resources initialized in async initialize()
        self.keywords = None
        self.escalation_reasons = []
    
    async def initialize(self) -> None:
        """Initialize escalation tool resources (async-safe)."""
        try:
            logger.info(f"Initializing Escalation tool '{self.name}'...")
            
            # Load custom keywords from settings
            self.keywords = ESCALATION_KEYWORDS.copy()
            
            # Add custom keywords if available
            if hasattr(settings, 'escalation_keywords'):
                custom_keywords = settings.escalation_keywords
                if isinstance(custom_keywords, dict):
                    self.keywords.update(custom_keywords)
                elif isinstance(custom_keywords, list):
                    for keyword in custom_keywords:
                        if keyword not in self.keywords:
                            self.keywords[keyword] = 0.8
            
            # Initialize escalation tracking
            self.escalation_reasons = []
            
            self.initialized = True
            logger.info(
                f"✓ Escalation tool '{self.name}' initialized "
                f"with {len(self.keywords)} keywords"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize Escalation tool: {e}", exc_info=True)
            raise
    
    async def cleanup(self) -> None:
        """Cleanup escalation tool resources."""
        try:
            logger.info(f"Cleaning up Escalation tool '{self.name}'...")
            
            self.escalation_reasons = []
            self.keywords = None
            
            self.initialized = False
            logger.info(f"✓ Escalation tool '{self.name}' cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during Escalation tool cleanup: {e}")
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute escalation check (async-first)."""
        message = kwargs.get("message")
        
        if not message:
            return ToolResult.error_result(
                error="message parameter is required",
                metadata={"tool": self.name}
            )
        
        try:
            # Perform escalation check
            result = await self.should_escalate_async(
                message=message,
                message_history=kwargs.get("message_history"),
                confidence_threshold=kwargs.get(
                    "confidence_threshold",
                    ESCALATION_CONFIDENCE_THRESHOLD
                ),
                metadata=kwargs.get("metadata")
            )
            
            # Create ticket if requested and escalation needed
            if result["escalate"] and kwargs.get("create_ticket", False):
                ticket = self.create_escalation_ticket(
                    session_id=kwargs.get("session_id", "unknown"),
                    escalation_result=result,
                    user_info=kwargs.get("user_info")
                )
                result["ticket"] = ticket
                
                # Send notification if configured
                if kwargs.get("notify", False):
                    notification = await self.notify_human_support_async(
                        ticket,
                        kwargs.get("notification_channel", "email")
                    )
                    result["notification"] = notification
            
            return ToolResult.success_result(
                data=result,
                metadata={
                    "tool": self.name,
                    "escalated": result["escalate"],
                    "confidence": result["confidence"],
                    "reasons_count": len(result.get("reasons", []))
                }
            )
            
        except Exception as e:
            logger.error(f"Escalation execute error: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={"tool": self.name, "message_preview": message[:100]}
            )
    
    async def should_escalate_async(
        self,
        message: str,
        message_history: Optional[List[Dict[str, Any]]] = None,
        confidence_threshold: float = ESCALATION_CONFIDENCE_THRESHOLD,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Determine if conversation should be escalated.
        
        Version 2.0.0: FIXED - Runs analysis directly without thread pool.
        """
        escalation_signals = []
        total_confidence = 0.0
        
        # FIXED: Run analysis directly (it's fast enough, no need for thread pool)
        analysis_result = self._analyze_message(
            message,
            message_history,
            metadata
        )
        
        # Unpack analysis results
        keyword_score = analysis_result['keyword_score']
        found_keywords = analysis_result['found_keywords']
        sentiment = analysis_result['sentiment']
        urgency = analysis_result['urgency']
        patterns = analysis_result['patterns']
        explicit_request = analysis_result['explicit_request']
        
        # 1. Check for escalation keywords
        if keyword_score > 0:
            escalation_signals.append(f"Keywords detected: {', '.join(found_keywords)}")
            total_confidence += keyword_score * 0.4  # 40% weight
        
        # 2. Analyze sentiment
        if sentiment < NEGATIVE_SENTIMENT_THRESHOLD:
            escalation_signals.append(f"Negative sentiment: {sentiment:.2f}")
            total_confidence += abs(sentiment) * 0.2  # 20% weight
        
        # 3. Check urgency
        if urgency > 0.5:
            escalation_signals.append(f"High urgency: {urgency:.2f}")
            total_confidence += urgency * 0.2  # 20% weight
        
        # 4. Analyze conversation patterns
        if patterns['repetitive_questions']:
            escalation_signals.append("Repetitive questions detected")
            total_confidence += 0.1
        
        if patterns['unresolved_issues']:
            escalation_signals.append("Long conversation without resolution")
            total_confidence += 0.1
        
        if patterns['degrading_sentiment']:
            escalation_signals.append("Degrading customer sentiment")
            total_confidence += 0.15
        
        if patterns['multiple_problems']:
            escalation_signals.append("Multiple issues reported")
            total_confidence += 0.1
        
        # 5. Check for explicit escalation request
        if explicit_request:
            escalation_signals.append("Explicit escalation request")
            total_confidence = 1.0  # Always escalate on explicit request
        
        # Determine if should escalate
        should_escalate = total_confidence >= confidence_threshold
        
        # Build response
        result = {
            "escalate": should_escalate,
            "confidence": min(total_confidence, 1.0),
            "reasons": escalation_signals,
            "urgency": urgency,
            "sentiment": sentiment,
            "threshold": confidence_threshold
        }
        
        # Add escalation category if escalating
        if should_escalate:
            if "legal" in message.lower() or "lawsuit" in message.lower():
                result["category"] = "legal"
                result["priority"] = "high"
            elif urgency > 0.7:
                result["category"] = "urgent"
                result["priority"] = "high"
            elif sentiment < -0.7:
                result["category"] = "complaint"
                result["priority"] = "medium"
            else:
                result["category"] = "general"
                result["priority"] = "normal"
        
        logger.info(
            f"Escalation check: {should_escalate} "
            f"(confidence: {total_confidence:.2f}, reasons: {len(escalation_signals)})"
        )
        
        return result
    
    async def notify_human_support_async(
        self,
        ticket: Dict[str, Any],
        notification_channel: str = "email"
    ) -> Dict[str, Any]:
        """Notify human support about escalation (async)."""
        # Simulate notification sending
        await asyncio.sleep(0.1)
        
        notification = {
            "channel": notification_channel,
            "ticket_id": ticket["ticket_id"],
            "sent_at": datetime.utcnow().isoformat(),
            "status": "sent"
        }
        
        if notification_channel == "email":
            logger.info(f"Email notification sent for ticket {ticket['ticket_id']}")
            notification["recipient"] = getattr(
                settings,
                'escalation_notification_email',
                'support@example.com'
            )
            
        elif notification_channel == "slack":
            logger.info(f"Slack notification sent for ticket {ticket['ticket_id']}")
            notification["channel_id"] = "#customer-support"
        
        return notification
    
    # ===========================
    # Private Helper Methods (Sync - Fast Operations)
    # ===========================
    
    def _analyze_message(
        self,
        message: str,
        message_history: Optional[List[Dict[str, Any]]],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze message for escalation signals (sync, CPU-bound but fast).
        
        Version 2.0.0: FIXED - Runs directly, no thread pool needed.
        """
        # Detect keywords
        keyword_score, found_keywords = self.detect_keywords(message)
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment(message)
        
        # Calculate urgency
        urgency = self.calculate_urgency_score(message, metadata)
        
        # Check conversation patterns
        patterns = self.check_conversation_patterns(message_history or [
