"""
Memory service for managing conversation memory and context.
Provides high-level operations over the memory tool.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json

from sqlalchemy import func, and_, or_, desc
from sqlalchemy.orm import Session

from ..config import settings
from ..models.memory import Memory
from ..tools.memory_tool import MemoryTool
from ..database import get_db

logger = logging.getLogger(__name__)


class MemoryService:
    """
    High-level service for memory management.
    Provides session-based memory operations, analytics, and insights.
    """
    
    def __init__(self):
        """Initialize memory service."""
        self.memory_tool = MemoryTool()
        self.enabled = settings.memory_enabled
        self.max_entries = settings.memory_max_entries
        self.ttl_hours = settings.memory_ttl_hours
        
        logger.info(f"Memory service initialized (enabled: {self.enabled})")
    
    async def create_session_memory(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Initialize memory for a new session.
        
        Args:
            session_id: Session identifier
            user_id: Optional user identifier
            metadata: Optional session metadata
            
        Returns:
            Initialization status
        """
        if not self.enabled:
            return {"success": False, "message": "Memory disabled"}
        
        try:
            # Store initial session context
            initial_context = f"New session started at {datetime.utcnow().isoformat()}"
            if user_id:
                initial_context += f" for user {user_id}"
            
            result = await self.memory_tool.store_memory(
                session_id=session_id,
                content=initial_context,
                content_type="context",
                metadata={
                    "user_id": user_id,
                    "session_metadata": metadata or {}
                },
                importance=0.3
            )
            
            logger.info(f"Created memory for session {session_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create session memory: {e}")
            return {"success": False, "error": str(e)}
    
    async def add_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        sources_used: Optional[List[str]] = None,
        confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Add a conversation turn to memory.
        
        Args:
            session_id: Session identifier
            user_message: User's message
            assistant_message: Assistant's response
            sources_used: Sources referenced in response
            confidence: Response confidence score
            
        Returns:
            Storage status
        """
        if not self.enabled:
            return {"success": False, "message": "Memory disabled"}
        
        results = []
        
        try:
            # Store user message
            user_result = await self.memory_tool.store_memory(
                session_id=session_id,
                content=f"User asked: {user_message[:500]}",
                content_type="context",
                metadata={"role": "user", "timestamp": datetime.utcnow().isoformat()},
                importance=0.6
            )
            results.append(user_result)
            
            # Store assistant response
            assistant_result = await self.memory_tool.store_memory(
                session_id=session_id,
                content=f"Assistant responded: {assistant_message[:500]}",
                content_type="context",
                metadata={
                    "role": "assistant",
                    "confidence": confidence,
                    "sources": sources_used or [],
                    "timestamp": datetime.utcnow().isoformat()
                },
                importance=0.5
            )
            results.append(assistant_result)
            
            # Extract and store important facts
            facts = await self._extract_facts(user_message, assistant_message)
            for fact in facts:
                fact_result = await self.memory_tool.store_memory(
                    session_id=session_id,
                    content=fact,
                    content_type="fact",
                    importance=0.8
                )
                results.append(fact_result)
            
            return {
                "success": all(r.get("success") for r in results),
                "stored_items": len(results),
                "facts_extracted": len(facts)
            }
            
        except Exception as e:
            logger.error(f"Failed to add conversation turn: {e}")
            return {"success": False, "error": str(e)}
    
    async def store_user_preference(
        self,
        session_id: str,
        preference: str,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store a user preference.
        
        Args:
            session_id: Session identifier
            preference: Preference description
            category: Optional preference category
            
        Returns:
            Storage status
        """
        if not self.enabled:
            return {"success": False, "message": "Memory disabled"}
        
        metadata = {"type": "preference"}
        if category:
            metadata["category"] = category
        
        return await self.memory_tool.store_memory(
            session_id=session_id,
            content=preference,
            content_type="preference",
            metadata=metadata,
            importance=0.9
        )
    
    async def store_user_info(
        self,
        session_id: str,
        info: str,
        info_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store user information.
        
        Args:
            session_id: Session identifier
            info: User information
            info_type: Type of information (name, email, etc.)
            
        Returns:
            Storage status
        """
        if not self.enabled:
            return {"success": False, "message": "Memory disabled"}
        
        metadata = {}
        if info_type:
            metadata["info_type"] = info_type
        
        return await self.memory_tool.store_memory(
            session_id=session_id,
            content=info,
            content_type="user_info",
            metadata=metadata,
            importance=1.0
        )
    
    async def get_session_context(
        self,
        session_id: str,
        include_facts: bool = True,
        include_preferences: bool = True,
        max_items: int = 20
    ) -> Dict[str, Any]:
        """
        Get comprehensive session context.
        
        Args:
            session_id: Session identifier
            include_facts: Include extracted facts
            include_preferences: Include user preferences
            max_items: Maximum items to retrieve
            
        Returns:
            Session context with categorized memories
        """
        if not self.enabled:
            return {"success": False, "message": "Memory disabled", "context": {}}
        
        try:
            context = {
                "user_info": [],
                "preferences": [],
                "facts": [],
                "recent_context": [],
                "summary": ""
            }
            
            # Get user information
            user_info = await self.memory_tool.retrieve_memories(
                session_id=session_id,
                content_type="user_info",
                limit=5
            )
            context["user_info"] = [m["content"] for m in user_info]
            
            # Get preferences
            if include_preferences:
                preferences = await self.memory_tool.retrieve_memories(
                    session_id=session_id,
                    content_type="preference",
                    limit=10
                )
                context["preferences"] = [m["content"] for m in preferences]
            
            # Get facts
            if include_facts:
                facts = await self.memory_tool.retrieve_memories(
                    session_id=session_id,
                    content_type="fact",
                    limit=10
                )
                context["facts"] = [m["content"] for m in facts]
            
            # Get recent context
            recent = await self.memory_tool.retrieve_memories(
                session_id=session_id,
                content_type="context",
                limit=max_items,
                time_window_hours=1
            )
            context["recent_context"] = [m["content"] for m in recent]
            
            # Generate summary
            context["summary"] = await self.memory_tool.summarize_session(session_id)
            
            return {
                "success": True,
                "session_id": session_id,
                "context": context,
                "total_items": sum(len(v) for v in context.values() if isinstance(v, list))
            }
            
        except Exception as e:
            logger.error(f"Failed to get session context: {e}")
            return {"success": False, "error": str(e), "context": {}}
    
    async def search_memories(
        self,
        session_id: str,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search memories by content.
        
        Args:
            session_id: Session identifier
            query: Search query
            limit: Maximum results
            
        Returns:
            Matching memories
        """
        if not self.enabled:
            return []
        
        try:
            with self.memory_tool.get_db() as db:
                # Search in memory content
                memories = db.query(Memory).filter(
                    and_(
                        Memory.session_id == session_id,
                        Memory.content.contains(query)
                    )
                ).order_by(
                    desc(Memory.importance),
                    desc(Memory.created_at)
                ).limit(limit).all()
                
                return [
                    {
                        "id": m.id,
                        "content": m.content,
                        "type": m.content_type,
                        "importance": m.importance,
                        "created_at": m.created_at.isoformat()
                    }
                    for m in memories
                ]
                
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []
    
    async def get_session_insights(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Get analytical insights about a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session insights and statistics
        """
        try:
            with self.memory_tool.get_db() as db:
                # Get memory statistics
                stats = db.query(
                    Memory.content_type,
                    func.count(Memory.id).label('count'),
                    func.avg(Memory.importance).label('avg_importance')
                ).filter(
                    Memory.session_id == session_id
                ).group_by(Memory.content_type).all()
                
                # Get time-based statistics
                time_stats = db.query(
                    func.min(Memory.created_at).label('first_memory'),
                    func.max(Memory.created_at).label('last_memory'),
                    func.count(Memory.id).label('total_memories')
                ).filter(
                    Memory.session_id == session_id
                ).first()
                
                # Calculate session duration
                duration = None
                if time_stats.first_memory and time_stats.last_memory:
                    duration = (time_stats.last_memory - time_stats.first_memory).total_seconds()
                
                # Get most important memories
                important = db.query(Memory).filter(
                    Memory.session_id == session_id
                ).order_by(
                    desc(Memory.importance)
                ).limit(5).all()
                
                return {
                    "session_id": session_id,
                    "statistics": {
                        "total_memories": time_stats.total_memories if time_stats else 0,
                        "duration_seconds": duration,
                        "memory_types": {
                            stat.content_type: {
                                "count": stat.count,
                                "avg_importance": float(stat.avg_importance) if stat.avg_importance else 0
                            }
                            for stat in stats
                        }
                    },
                    "important_memories": [
                        {
                            "content": m.content,
                            "type": m.content_type,
                            "importance": m.importance
                        }
                        for m in important
                    ],
                    "timeline": {
                        "first_activity": time_stats.first_memory.isoformat() if time_stats and time_stats.first_memory else None,
                        "last_activity": time_stats.last_memory.isoformat() if time_stats and time_stats.last_memory else None
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get session insights: {e}")
            return {"error": str(e)}
    
    async def merge_sessions(
        self,
        source_session_id: str,
        target_session_id: str
    ) -> Dict[str, Any]:
        """
        Merge memories from one session to another.
        
        Args:
            source_session_id: Source session to merge from
            target_session_id: Target session to merge into
            
        Returns:
            Merge operation status
        """
        if not self.enabled:
            return {"success": False, "message": "Memory disabled"}
        
        try:
            with self.memory_tool.get_db() as db:
                # Update all memories from source to target
                updated = db.query(Memory).filter(
                    Memory.session_id == source_session_id
                ).update(
                    {Memory.session_id: target_session_id},
                    synchronize_session=False
                )
                
                db.commit()
                
                logger.info(f"Merged {updated} memories from {source_session_id} to {target_session_id}")
                
                return {
                    "success": True,
                    "memories_merged": updated,
                    "source_session": source_session_id,
                    "target_session": target_session_id
                }
                
        except Exception as e:
            logger.error(f"Failed to merge sessions: {e}")
            return {"success": False, "error": str(e)}
    
    async def export_session_memory(
        self,
        session_id: str,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Export session memory in specified format.
        
        Args:
            session_id: Session identifier
            format: Export format (json, text)
            
        Returns:
            Exported memory data
        """
        try:
            # Get all memories
            memories = await self.memory_tool.retrieve_memories(
                session_id=session_id,
                limit=1000
            )
            
            if format == "json":
                return {
                    "session_id": session_id,
                    "export_date": datetime.utcnow().isoformat(),
                    "memories": memories,
                    "total_count": len(memories)
                }
            
            elif format == "text":
                # Format as readable text
                text_lines = [
                    f"Session Memory Export: {session_id}",
                    f"Export Date: {datetime.utcnow().isoformat()}",
                    "=" * 50,
                    ""
                ]
                
                # Group by type
                by_type = defaultdict(list)
                for mem in memories:
                    by_type[mem["content_type"]].append(mem)
                
                for content_type, items in by_type.items():
                    text_lines.append(f"\n{content_type.upper()}:")
                    text_lines.append("-" * 30)
                    for item in items:
                        text_lines.append(f"• {item['content']}")
                        text_lines.append(f"  (Importance: {item['importance']}, Time: {item['created_at']})")
                
                return {
                    "session_id": session_id,
                    "format": "text",
                    "content": "\n".join(text_lines)
                }
            
            else:
                return {"error": f"Unsupported format: {format}"}
                
        except Exception as e:
            logger.error(f"Failed to export session memory: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_sessions(
        self,
        days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Clean up old session memories.
        
        Args:
            days: Days to keep (defaults to config)
            
        Returns:
            Cleanup statistics
        """
        days = days or settings.memory_cleanup_days
        
        try:
            result = await self.memory_tool.cleanup_old_memories(days=days)
            
            logger.info(f"Memory cleanup completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            return {"success": False, "error": str(e)}
    
    async def _extract_facts(
        self,
        user_message: str,
        assistant_message: str
    ) -> List[str]:
        """
        Extract important facts from conversation.
        
        Args:
            user_message: User's message
            assistant_message: Assistant's response
            
        Returns:
            List of extracted facts
        """
        facts = []
        
        # Look for patterns in user message
        import re
        
        # Email pattern
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', user_message)
        for email in emails:
            facts.append(f"User email: {email}")
        
        # Phone pattern
        phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', user_message)
        for phone in phones:
            facts.append(f"User phone: {phone}")
        
        # Order/reference numbers
        refs = re.findall(r'\b(?:order|ticket|ref|reference|confirmation)[\s#]*([A-Z0-9-]+)\b', user_message, re.I)
        for ref in refs:
            facts.append(f"Reference: {ref}")
        
        # Name patterns (simple heuristic)
        if "my name is" in user_message.lower():
            name_match = re.search(r'my name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', user_message, re.I)
            if name_match:
                facts.append(f"User name: {name_match.group(1)}")
        
        return facts[:5]  # Limit to 5 facts per turn
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory service statistics.
        
        Returns:
            Service statistics
        """
        try:
            with self.memory_tool.get_db() as db:
                total = db.query(func.count(Memory.id)).scalar()
                sessions = db.query(func.count(func.distinct(Memory.session_id))).scalar()
                
                return {
                    "enabled": self.enabled,
                    "total_memories": total,
                    "active_sessions": sessions,
                    "max_entries_per_session": self.max_entries,
                    "ttl_hours": self.ttl_hours
                }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "enabled": self.enabled,
                "error": str(e)
            }


# Global memory service instance
memory_service = MemoryService()

# Export for convenience
__all__ = ['MemoryService', 'memory_service']
