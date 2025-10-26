"""
Message model for storing conversation messages.
"""
from sqlalchemy import Column, String, Text, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class Message(Base):
    """
    Chat message model.
    """
    __tablename__ = "messages"
    
    id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=False, index=True)
    
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Optional fields
    attachments = Column(JSON, default=list)  # List of attachment info
    sources = Column(JSON, default=list)  # RAG sources used
    metadata = Column(JSON, default=dict)  # Additional metadata
    
    # Tool usage tracking
    tools_used = Column(JSON, default=list)
    processing_time = Column(Float, nullable=True)  # Time taken to generate response
    
    def __repr__(self):
        return f"<Message(id={self.id}, session={self.session_id}, role={self.role})>"
