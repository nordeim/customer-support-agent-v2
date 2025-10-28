"""
Session model for storing chat sessions.
"""
from sqlalchemy import Column, String, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class Session(Base):
    """
    Chat session model.
    """
    __tablename__ = "sessions"
    
    id = Column(String(36), primary_key=True)
    user_id = Column(String(100), nullable=True, index=True)
    thread_id = Column(String(36), nullable=True)  # Agent framework thread ID
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    
    status = Column(String(20), default="active")  # active, ended, escalated
    escalated = Column(Boolean, default=False)
    escalation_ticket_id = Column(String(50), nullable=True)
    
    tool_metadata = Column("metadata", JSON, default=dict)
    
    def __repr__(self):
        return f"<Session(id={self.id}, user={self.user_id}, status={self.status})>"
