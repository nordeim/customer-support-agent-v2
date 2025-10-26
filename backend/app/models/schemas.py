"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Message role enumeration."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SessionStatus(str, Enum):
    """Session status enumeration."""
    ACTIVE = "active"
    ENDED = "ended"
    ESCALATED = "escalated"


# Request Schemas

class CreateSessionRequest(BaseModel):
    """Request to create a new session."""
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "metadata": {"source": "web", "language": "en"}
            }
        }


class SendMessageRequest(BaseModel):
    """Request to send a message."""
    message: str = Field(..., min_length=1, max_length=4000)
    attachments: Optional[List[Dict[str, Any]]] = []
    
    @validator('message')
    def validate_message(cls, v):
        """Ensure message is not just whitespace."""
        if not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "message": "I need help with my order",
                "attachments": []
            }
        }


class SearchRequest(BaseModel):
    """Request to search knowledge base."""
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(5, ge=1, le=20)
    filters: Optional[Dict[str, Any]] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "query": "refund policy",
                "limit": 5,
                "filters": {"category": "policies"}
            }
        }


# Response Schemas

class SessionResponse(BaseModel):
    """Session information response."""
    session_id: str
    user_id: Optional[str]
    thread_id: Optional[str]
    status: SessionStatus
    created_at: datetime
    metadata: Dict[str, Any] = {}
    
    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "session_id": "sess_abc123",
                "user_id": "user123",
                "thread_id": "thread_xyz",
                "status": "active",
                "created_at": "2024-01-15T10:30:00Z",
                "metadata": {}
            }
        }


class SourceInfo(BaseModel):
    """Information about a knowledge source."""
    content: str
    metadata: Dict[str, Any] = {}
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    rank: Optional[int] = None
    
    class Config:
        schema_extra = {
            "example": {
                "content": "Our refund policy allows returns within 30 days...",
                "metadata": {"source": "policy.pdf", "page": 5},
                "relevance_score": 0.95,
                "rank": 1
            }
        }


class ChatResponse(BaseModel):
    """Chat message response."""
    message: str
    sources: List[SourceInfo] = []
    requires_escalation: bool = False
    confidence: float = Field(0.95, ge=0.0, le=1.0)
    session_id: str
    timestamp: datetime
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "message": "I can help you with that...",
                "sources": [],
                "requires_escalation": False,
                "confidence": 0.95,
                "session_id": "sess_abc123",
                "timestamp": "2024-01-15T10:30:00Z",
                "processing_time": 1.5,
                "metadata": {}
            }
        }


class MessageHistory(BaseModel):
    """Message history response."""
    messages: List[Dict[str, Any]]
    total: int
    session_id: str
    
    class Config:
        schema_extra = {
            "example": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello",
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                ],
                "total": 1,
                "session_id": "sess_abc123"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, str] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "1.0.0",
                "services": {
                    "database": "healthy",
                    "redis": "healthy",
                    "chromadb": "healthy"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    message: str
    request_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid request parameters",
                "request_id": "req_123",
                "details": {"field": "message", "issue": "too long"}
            }
        }


# WebSocket Schemas

class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str  # "message", "status", "error", "complete"
    content: Optional[str] = None
    data: Optional[Dict[str, Any]] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "type": "message",
                "content": "Hello, how can I help?",
                "data": {},
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class FileUploadResponse(BaseModel):
    """File upload response."""
    filename: str
    size_mb: float
    content_type: str
    processed: bool
    preview: Optional[str] = None
    error: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "document.pdf",
                "size_mb": 1.5,
                "content_type": "application/pdf",
                "processed": True,
                "preview": "This document contains...",
                "error": None
            }
        }
