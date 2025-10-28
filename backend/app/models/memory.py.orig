"""
SQLAlchemy models for memory storage.
"""
from sqlalchemy import Column, String, Text, DateTime, Float, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class Memory(Base):
    """
    Memory storage model for conversation context and facts.
    """
    __tablename__ = "memories"
    
    # Primary key
    id = Column(String(36), primary_key=True)
    
    # Session association
    session_id = Column(String(36), nullable=False, index=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    
    # Content
    content_type = Column(String(50), nullable=False)  # 'user_info', 'preference', 'context', 'fact'
    content = Column(Text, nullable=False)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    
    # Importance scoring
    importance = Column(Float, default=0.5, nullable=False)  # 0.0 to 1.0
    access_count = Column(Integer, default=0)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_session_type', 'session_id', 'content_type'),
        Index('idx_session_importance', 'session_id', 'importance'),
        Index('idx_last_accessed', 'last_accessed'),
    )
    
    def __repr__(self):
        return f"<Memory(id={self.id}, session={self.session_id}, type={self.content_type})>"
