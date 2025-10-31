# backend/app/models/memory.py
```py
"""
Memory model for conversation context persistence.
Stores important information extracted from conversations.

Version: 2.0.0 (Enhanced with unique constraints and indexes)
"""
from sqlalchemy import (
    Column, String, Text, Float, Integer, DateTime, JSON,
    Index, UniqueConstraint
)
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, Any, Optional
import json

from ..database import Base


class Memory(Base):
    """
    Memory storage for conversation context.
    
    Stores important facts, preferences, and context extracted
    from user conversations for enhanced personalization.
    
    Version 2.0.0:
    - Added unique constraint to prevent duplicates
    - Added composite index for efficient queries
    - Fixed field naming (metadata instead of tool_metadata)
    """
    
    __tablename__ = "memories"
    
    # Primary key
    id = Column(String(36), primary_key=True, index=True)
    
    # Session reference
    session_id = Column(String(255), nullable=False, index=True)
    
    # Memory classification
    content_type = Column(
        String(50),
        nullable=False,
        default="context",
        comment="Type: user_info, preference, fact, context"
    )
    
    # Memory content
    content = Column(Text, nullable=False)
    
    # FIXED: Renamed from tool_metadata to metadata for consistency
    metadata = Column(
        JSON,
        nullable=False,
        default=dict,
        comment="Additional metadata about the memory"
    )
    
    # Importance score (0.0 to 1.0)
    importance = Column(
        Float,
        nullable=False,
        default=0.5,
        comment="Importance score for retrieval prioritization"
    )
    
    # Access tracking
    access_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of times this memory was accessed"
    )
    
    last_accessed = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last time this memory was retrieved"
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Table-level constraints and indexes
    __table_args__ = (
        # CRITICAL FIX: Unique constraint to prevent duplicate memories
        UniqueConstraint(
            'session_id',
            'content_type',
            'content',
            name='uq_memory_session_content'
        ),
        
        # Composite index for efficient retrieval queries
        Index(
            'ix_memory_session_type_importance',
            'session_id',
            'content_type',
            'importance'
        ),
        
        # Index for cleanup queries
        Index(
            'ix_memory_last_accessed',
            'last_accessed'
        ),
        
        # Index for time-based queries
        Index(
            'ix_memory_created_at',
            'created_at'
        ),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<Memory(id={self.id}, session_id={self.session_id}, "
            f"type={self.content_type}, importance={self.importance})>"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "session_id": self.session_id,
            "content_type": self.content_type,
            "content": self.content,
            "metadata": self.metadata,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """
        Create Memory instance from dictionary.
        
        Args:
            data: Dictionary with memory data
            
        Returns:
            Memory instance
        """
        # Convert ISO strings to datetime if needed
        for field in ['last_accessed', 'created_at', 'updated_at']:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)
    
    def update_access(self) -> None:
        """Update access tracking."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
    
    def validate(self) -> bool:
        """
        Validate memory data.
        
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        # Validate content_type
        valid_types = ['user_info', 'preference', 'fact', 'context']
        if self.content_type not in valid_types:
            raise ValueError(f"Invalid content_type. Must be one of: {valid_types}")
        
        # Validate importance range
        if not (0.0 <= self.importance <= 1.0):
            raise ValueError("Importance must be between 0.0 and 1.0")
        
        # Validate content not empty
        if not self.content or not self.content.strip():
            raise ValueError("Content cannot be empty")
        
        # Validate metadata is JSON-serializable
        try:
            json.dumps(self.metadata)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Metadata must be JSON-serializable: {e}")
        
        return True


__all__ = ['Memory']

```

# backend/app/models/__init__.py
```py
"""
Database models package.
Exports all SQLAlchemy models for the application.

Version: 2.0.0
"""

from .memory import Memory
from .session import Session
from .message import Message

__all__ = [
    'Memory',
    'Session',
    'Message'
]

```

