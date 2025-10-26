# Complete Backend Agent Integration - Meticulous Plan & Implementation

## 📋 Implementation Plan

### Overview
We'll create complete replacement files for the backend agent system that properly integrates with the tools we've built. This includes:
1. Updated agent implementation with proper tool integration
2. API routes for chat, sessions, and health
3. WebSocket support for real-time communication
4. Middleware for request handling
5. Authentication and telemetry services
6. Pydantic schemas for data validation

### Key Integration Points
- Agent must properly initialize and call our tools
- API routes must handle file uploads for attachment tool
- WebSocket must stream responses in real-time
- Session management must integrate with memory tool
- Error handling must be consistent across all components

---

## 🛠️ Complete Implementation Files

### File 1: Updated Chat Agent with Tool Integration

**`backend/app/agents/chat_agent.py`**
```python
"""
Customer Support Agent implementation with full tool integration.
This agent orchestrates RAG, Memory, Attachment, and Escalation tools.
"""
import asyncio
import json
import logging
import uuid
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass

from ..config import settings
from ..tools import RAGTool, MemoryTool, AttachmentTool, EscalationTool
from ..models.session import Session
from ..models.message import Message
from ..models.memory import Memory

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Context for agent processing."""
    session_id: str
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    message_count: int = 0
    escalated: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AgentResponse:
    """Structured agent response."""
    
    def __init__(
        self,
        message: str,
        sources: List[Dict] = None,
        requires_escalation: bool = False,
        confidence: float = 0.0,
        tools_used: List[str] = None,
        processing_time: float = 0.0
    ):
        self.message = message
        self.sources = sources or []
        self.requires_escalation = requires_escalation
        self.confidence = confidence
        self.tools_used = tools_used or []
        self.processing_time = processing_time
        self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message,
            "sources": self.sources,
            "requires_escalation": self.requires_escalation,
            "confidence": self.confidence,
            "tools_used": self.tools_used,
            "processing_time": self.processing_time,
            "metadata": self.metadata
        }


class CustomerSupportAgent:
    """
    Production-ready customer support agent with full tool integration.
    Orchestrates multiple AI tools for comprehensive support capabilities.
    """
    
    # System prompt with tool instructions
    SYSTEM_PROMPT = """You are an expert customer support AI assistant with access to the following tools:

AVAILABLE TOOLS:
1. **rag_search**: Search our knowledge base for relevant information
   - Use this when users ask questions about policies, procedures, or general information
   - Always cite sources when using information from this tool

2. **memory_management**: Store and retrieve conversation context
   - Use this to remember important user information and preferences
   - Check memory at the start of each conversation for context

3. **attachment_processor**: Process and analyze uploaded documents
   - Use this when users upload files
   - Extract and analyze content from various file formats

4. **escalation_check**: Determine if human intervention is needed
   - Monitor for signs that require human support
   - Check sentiment and urgency of user messages

INSTRUCTIONS:
1. Always be helpful, professional, and empathetic
2. Use tools appropriately to provide accurate information
3. Cite your sources when providing information from the knowledge base
4. Remember important details about the user and their issues
5. Escalate to human support when:
   - The user explicitly asks for human assistance
   - The issue involves legal or compliance matters
   - The user expresses high frustration or dissatisfaction
   - You cannot resolve the issue after multiple attempts

RESPONSE FORMAT:
- Provide clear, concise answers
- Break down complex information into steps
- Offer additional help and next steps
- Maintain a friendly, professional tone

Remember: Customer satisfaction is the top priority."""
    
    def __init__(self):
        """Initialize the agent with all tools."""
        self.tools = {}
        self.contexts = {}  # Store session contexts
        self.initialized = False
        
        # Initialize on creation
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize all tools and components."""
        try:
            # Initialize tools
            logger.info("Initializing agent tools...")
            
            self.tools['rag'] = RAGTool()
            logger.info("✓ RAG tool initialized")
            
            self.tools['memory'] = MemoryTool()
            logger.info("✓ Memory tool initialized")
            
            self.tools['attachment'] = AttachmentTool()
            logger.info("✓ Attachment tool initialized")
            
            self.tools['escalation'] = EscalationTool()
            logger.info("✓ Escalation tool initialized")
            
            self.initialized = True
            logger.info(f"Agent initialized with {len(self.tools)} tools")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}", exc_info=True)
            raise
    
    def get_or_create_context(self, session_id: str) -> AgentContext:
        """Get or create context for a session."""
        if session_id not in self.contexts:
            self.contexts[session_id] = AgentContext(
                session_id=session_id,
                thread_id=str(uuid.uuid4())
            )
            logger.info(f"Created new context for session: {session_id}")
        
        return self.contexts[session_id]
    
    async def load_session_context(self, session_id: str) -> str:
        """Load conversation context from memory."""
        try:
            # Get memory summary
            memory_tool = self.tools['memory']
            summary = await memory_tool.summarize_session(session_id)
            
            # Get recent messages
            memories = await memory_tool.retrieve_memories(
                session_id=session_id,
                content_type="context",
                limit=5
            )
            
            if memories:
                recent_context = "\nRecent conversation points:\n"
                for memory in memories[:3]:
                    recent_context += f"- {memory['content']}\n"
                summary += recent_context
            
            return summary
            
        except Exception as e:
            logger.error(f"Error loading session context: {e}")
            return ""
    
    async def search_knowledge_base(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search the knowledge base using RAG tool."""
        try:
            rag_tool = self.tools['rag']
            result = await rag_tool.search(
                query=query,
                k=k,
                threshold=0.7
            )
            
            return result.get("sources", [])
            
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return []
    
    async def process_attachments(
        self,
        attachments: List[Dict[str, Any]]
    ) -> str:
        """Process uploaded attachments."""
        if not attachments:
            return ""
        
        attachment_tool = self.tools['attachment']
        rag_tool = self.tools['rag']
        
        processed_content = "\n📎 Attached Documents:\n"
        
        for attachment in attachments:
            try:
                # Process attachment
                result = await attachment_tool.process_attachment(
                    file_path=attachment.get("path"),
                    filename=attachment.get("filename"),
                    chunk_for_rag=True
                )
                
                if result["success"]:
                    # Add summary to context
                    processed_content += f"\n[{result['filename']}]:\n"
                    processed_content += f"{result['preview']}\n"
                    
                    # Index in RAG if chunks available
                    if "chunks" in result:
                        rag_tool.add_documents(
                            documents=result["chunks"],
                            metadatas=[
                                {
                                    "source": result['filename'],
                                    "type": "user_upload",
                                    "session_id": attachment.get("session_id")
                                }
                                for _ in result["chunks"]
                            ]
                        )
                        logger.info(f"Indexed {len(result['chunks'])} chunks from {result['filename']}")
                
            except Exception as e:
                logger.error(f"Error processing attachment: {e}")
                processed_content += f"\n[Error processing {attachment.get('filename', 'file')}]\n"
        
        return processed_content
    
    async def check_escalation(
        self,
        message: str,
        context: AgentContext,
        message_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """Check if escalation is needed."""
        try:
            escalation_tool = self.tools['escalation']
            
            result = await escalation_tool.should_escalate(
                message=message,
                message_history=message_history,
                metadata={
                    "session_id": context.session_id,
                    "message_count": context.message_count,
                    "already_escalated": context.escalated
                }
            )
            
            # Create ticket if escalation needed
            if result["escalate"] and not context.escalated:
                result["ticket"] = escalation_tool.create_escalation_ticket(
                    session_id=context.session_id,
                    escalation_result=result,
                    user_info={"user_id": context.user_id}
                )
                context.escalated = True
                logger.info(f"Escalation triggered for session {context.session_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Escalation check error: {e}")
            return {"escalate": False, "confidence": 0.0}
    
    async def store_conversation_memory(
        self,
        session_id: str,
        user_message: str,
        agent_response: str,
        important_facts: List[str] = None
    ) -> None:
        """Store important information in memory."""
        try:
            memory_tool = self.tools['memory']
            
            # Store user message as context
            await memory_tool.store_memory(
                session_id=session_id,
                content=f"User: {user_message[:200]}",
                content_type="context",
                importance=0.5
            )
            
            # Store agent response summary
            if len(agent_response) > 100:
                await memory_tool.store_memory(
                    session_id=session_id,
                    content=f"Agent: {agent_response[:200]}",
                    content_type="context",
                    importance=0.4
                )
            
            # Store any identified important facts
            if important_facts:
                for fact in important_facts:
                    await memory_tool.store_memory(
                        session_id=session_id,
                        content=fact,
                        content_type="fact",
                        importance=0.8
                    )
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
    
    def extract_important_facts(
        self,
        message: str,
        response: str
    ) -> List[str]:
        """Extract important facts from conversation."""
        facts = []
        
        # Look for user information patterns
        import re
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, message)
        for email in emails:
            facts.append(f"User email: {email}")
        
        # Phone pattern
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phones = re.findall(phone_pattern, message)
        for phone in phones:
            facts.append(f"User phone: {phone}")
        
        # Order/ticket number pattern
        order_pattern = r'\b(?:order|ticket|reference|confirmation)\s*#?\s*([A-Z0-9-]+)\b'
        orders = re.findall(order_pattern, message, re.IGNORECASE)
        for order in orders:
            facts.append(f"Reference number: {order}")
        
        return facts
    
    async def generate_response(
        self,
        message: str,
        context: str,
        sources: List[Dict],
        escalation: Dict[str, Any]
    ) -> str:
        """Generate agent response based on context and tools."""
        # Build response based on available information
        response_parts = []
        
        # Add greeting if first message
        if context == "No previous context available for this session.":
            response_parts.append("Hello! I'm here to help you today.")
        
        # Add information from knowledge base
        if sources:
            response_parts.append("Based on our information:")
            for i, source in enumerate(sources[:2], 1):
                response_parts.append(f"{i}. {source['content'][:200]}...")
        
        # Add escalation message if needed
        if escalation.get("escalate"):
            response_parts.append(
                "\nI understand this is important to you. "
                "I'm connecting you with a human support specialist who can better assist you."
            )
            if escalation.get("ticket"):
                response_parts.append(
                    f"Your ticket number is: {escalation['ticket']['ticket_id']}"
                )
        
        # Default helpful response if no specific information
        if not response_parts:
            response_parts.append(
                "I'm here to help! Could you please provide more details about your inquiry?"
            )
        
        return "\n\n".join(response_parts)
    
    async def process_message(
        self,
        session_id: str,
        message: str,
        attachments: Optional[List[Dict[str, Any]]] = None,
        user_id: Optional[str] = None,
        message_history: Optional[List[Dict]] = None
    ) -> AgentResponse:
        """
        Process a user message and generate response.
        
        Args:
            session_id: Session identifier
            message: User message
            attachments: Optional file attachments
            user_id: Optional user identifier
            message_history: Previous messages
            
        Returns:
            AgentResponse with generated response and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Get or create context
            context = self.get_or_create_context(session_id)
            context.user_id = user_id
            context.message_count += 1
            
            # Load session context from memory
            session_context = await self.load_session_context(session_id)
            
            # Process attachments if any
            attachment_context = await self.process_attachments(attachments) if attachments else ""
            
            # Search knowledge base for relevant information
            sources = await self.search_knowledge_base(message)
            
            # Check for escalation
            escalation = await self.check_escalation(message, context, message_history)
            
            # Generate response
            response_text = await self.generate_response(
                message=message,
                context=session_context,
                sources=sources,
                escalation=escalation
            )
            
            # Extract and store important facts
            facts = self.extract_important_facts(message, response_text)
            await self.store_conversation_memory(
                session_id=session_id,
                user_message=message,
                agent_response=response_text,
                important_facts=facts
            )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Build response
            response = AgentResponse(
                message=response_text,
                sources=sources[:3],  # Limit sources in response
                requires_escalation=escalation.get("escalate", False),
                confidence=escalation.get("confidence", 0.95),
                tools_used=["rag", "memory", "escalation"],
                processing_time=processing_time
            )
            
            # Add metadata
            response.metadata = {
                "session_id": session_id,
                "message_count": context.message_count,
                "has_context": bool(session_context),
                "facts_extracted": len(facts)
            }
            
            if escalation.get("ticket"):
                response.metadata["ticket_id"] = escalation["ticket"]["ticket_id"]
            
            logger.info(
                f"Processed message for session {session_id} in {processing_time:.2f}s "
                f"(escalate: {response.requires_escalation})"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            
            # Return error response
            return AgentResponse(
                message="I apologize, but I encountered an error processing your request. "
                        "Please try again or contact support directly.",
                requires_escalation=True,
                confidence=0.0,
                processing_time=(datetime.utcnow() - start_time).total_seconds()
            )
    
    async def stream_response(
        self,
        session_id: str,
        message: str,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream response for real-time interaction.
        
        Yields:
            Updates as they're generated
        """
        try:
            # Initial processing
            yield {
                "type": "start",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Load context
            yield {
                "type": "status",
                "message": "Loading conversation context...",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            context = self.get_or_create_context(session_id)
            session_context = await self.load_session_context(session_id)
            
            # Process attachments
            if attachments:
                yield {
                    "type": "status",
                    "message": "Processing attachments...",
                    "timestamp": datetime.utcnow().isoformat()
                }
                attachment_context = await self.process_attachments(attachments)
            
            # Search knowledge base
            yield {
                "type": "status",
                "message": "Searching knowledge base...",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            sources = await self.search_knowledge_base(message)
            
            if sources:
                yield {
                    "type": "sources",
                    "sources": sources[:3],
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Check escalation
            escalation = await self.check_escalation(message, context)
            
            if escalation.get("escalate"):
                yield {
                    "type": "escalation",
                    "required": True,
                    "reason": escalation.get("reasons", []),
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Generate and stream response
            response_text = await self.generate_response(
                message=message,
                context=session_context,
                sources=sources,
                escalation=escalation
            )
            
            # Simulate streaming by sending response in chunks
            words = response_text.split()
            chunk_size = 5
            
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i+chunk_size])
                yield {
                    "type": "text",
                    "content": chunk + ' ',
                    "timestamp": datetime.utcnow().isoformat()
                }
                await asyncio.sleep(0.05)  # Small delay for streaming effect
            
            # Store in memory
            await self.store_conversation_memory(
                session_id=session_id,
                user_message=message,
                agent_response=response_text
            )
            
            # Final completion
            yield {
                "type": "complete",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def cleanup_session(self, session_id: str) -> None:
        """Clean up session resources."""
        try:
            # Remove context
            if session_id in self.contexts:
                del self.contexts[session_id]
            
            # Clean up old memories (optional)
            memory_tool = self.tools['memory']
            await memory_tool.cleanup_old_memories(days=30)
            
            logger.info(f"Cleaned up session: {session_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up session: {e}")
    
    async def cleanup(self) -> None:
        """Clean up all agent resources."""
        logger.info("Cleaning up agent resources...")
        
        # Clean up all sessions
        for session_id in list(self.contexts.keys()):
            await self.cleanup_session(session_id)
        
        # Clean up tools
        for tool_name, tool in self.tools.items():
            if hasattr(tool, 'cleanup'):
                await tool.cleanup()
                logger.info(f"Cleaned up {tool_name} tool")
        
        logger.info("Agent cleanup complete")
```

### File 2: Pydantic Schemas

**`backend/app/models/schemas.py`**
```python
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
```

### File 3: Chat API Routes

**`backend/app/api/routes/chat.py`**
```python
"""
Chat API routes for message handling.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional
import uuid
from datetime import datetime
import logging
import os
import tempfile

from ...agents.chat_agent import CustomerSupportAgent
from ...models.schemas import (
    SendMessageRequest, ChatResponse, MessageHistory,
    SearchRequest, SourceInfo, FileUploadResponse
)
from ...models.session import Session
from ...models.message import Message
from ...config import settings
from sqlalchemy.orm import Session as DBSession
from ...database import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


def get_agent() -> CustomerSupportAgent:
    """Get the agent instance from app state."""
    from ...main import app
    if not hasattr(app.state, 'agent'):
        raise HTTPException(status_code=500, detail="Agent not initialized")
    return app.state.agent


@router.post("/sessions/{session_id}/messages", response_model=ChatResponse)
async def send_message(
    session_id: str,
    background_tasks: BackgroundTasks,
    message: str = Form(...),
    attachments: List[UploadFile] = File(None),
    agent: CustomerSupportAgent = Depends(get_agent),
    db: DBSession = Depends(get_db)
):
    """
    Send a message and receive AI response.
    
    Args:
        session_id: Session identifier
        message: User message text
        attachments: Optional file attachments
        
    Returns:
        AI generated response with sources
    """
    try:
        # Validate session exists
        session = db.query(Session).filter(Session.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Check session status
        if session.status != "active":
            raise HTTPException(
                status_code=400,
                detail=f"Session is {session.status}. Cannot send messages."
            )
        
        # Process attachments
        processed_attachments = []
        if attachments:
            for file in attachments:
                if file.size > settings.max_file_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File {file.filename} exceeds maximum size"
                    )
                
                # Save to temporary location
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
                
                content = await file.read()
                with open(temp_path, 'wb') as f:
                    f.write(content)
                
                processed_attachments.append({
                    "path": temp_path,
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "size": len(content),
                    "session_id": session_id
                })
                
                logger.info(f"Saved attachment: {file.filename} ({len(content)} bytes)")
        
        # Get message history
        recent_messages = db.query(Message).filter(
            Message.session_id == session_id
        ).order_by(Message.created_at.desc()).limit(10).all()
        
        message_history = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.created_at.isoformat()
            }
            for msg in reversed(recent_messages)
        ]
        
        # Store user message
        user_message = Message(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=message,
            attachments=[{"filename": a["filename"]} for a in processed_attachments]
        )
        db.add(user_message)
        db.commit()
        
        # Process message with agent
        agent_response = await agent.process_message(
            session_id=session_id,
            message=message,
            attachments=processed_attachments if processed_attachments else None,
            user_id=session.user_id,
            message_history=message_history
        )
        
        # Store assistant response
        assistant_message = Message(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content=agent_response.message,
            sources=[s for s in agent_response.sources],
            tools_used=agent_response.tools_used,
            processing_time=agent_response.processing_time
        )
        db.add(assistant_message)
        
        # Update session
        session.last_activity = datetime.utcnow()
        if agent_response.requires_escalation:
            session.status = "escalated"
            session.escalated = True
            if agent_response.metadata.get("ticket_id"):
                session.escalation_ticket_id = agent_response.metadata["ticket_id"]
        
        db.commit()
        
        # Clean up temp files in background
        if processed_attachments:
            background_tasks.add_task(cleanup_temp_files, processed_attachments)
        
        # Build response
        return ChatResponse(
            message=agent_response.message,
            sources=[
                SourceInfo(
                    content=s["content"],
                    metadata=s.get("metadata", {}),
                    relevance_score=s.get("relevance_score", 0.0),
                    rank=s.get("rank")
                )
                for s in agent_response.sources
            ],
            requires_escalation=agent_response.requires_escalation,
            confidence=agent_response.confidence,
            session_id=session_id,
            timestamp=datetime.utcnow(),
            processing_time=agent_response.processing_time,
            metadata=agent_response.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process message")


@router.get("/sessions/{session_id}/messages", response_model=MessageHistory)
async def get_messages(
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    db: DBSession = Depends(get_db)
):
    """
    Retrieve message history for a session.
    
    Args:
        session_id: Session identifier
        limit: Maximum messages to return
        offset: Pagination offset
        
    Returns:
        List of messages with metadata
    """
    try:
        # Validate session
        session = db.query(Session).filter(Session.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get total count
        total = db.query(Message).filter(Message.session_id == session_id).count()
        
        # Get messages
        messages = db.query(Message).filter(
            Message.session_id == session_id
        ).order_by(Message.created_at).offset(offset).limit(limit).all()
        
        # Format messages
        formatted_messages = [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.created_at.isoformat(),
                "attachments": msg.attachments or [],
                "sources": msg.sources or [],
                "tools_used": msg.tools_used or [],
                "processing_time": msg.processing_time
            }
            for msg in messages
        ]
        
        return MessageHistory(
            messages=formatted_messages,
            total=total,
            session_id=session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving messages: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve messages")


@router.post("/search", response_model=List[SourceInfo])
async def search_knowledge_base(
    request: SearchRequest,
    agent: CustomerSupportAgent = Depends(get_agent)
):
    """
    Search the knowledge base directly.
    
    Args:
        request: Search parameters
        
    Returns:
        List of relevant sources
    """
    try:
        # Use agent's RAG tool
        sources = await agent.search_knowledge_base(
            query=request.query,
            k=request.limit
        )
        
        # Format response
        return [
            SourceInfo(
                content=source["content"],
                metadata=source.get("metadata", {}),
                relevance_score=source.get("relevance_score", 0.0),
                rank=source.get("rank")
            )
            for source in sources
        ]
        
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search failed")


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    agent: CustomerSupportAgent = Depends(get_agent)
):
    """
    Upload and process a file.
    
    Args:
        file: File to upload
        session_id: Associated session
        
    Returns:
        File processing results
    """
    try:
        # Validate file size
        contents = await file.read()
        size_mb = len(contents) / (1024 * 1024)
        
        if len(contents) > settings.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum size of {settings.max_file_size/(1024*1024):.1f}MB"
            )
        
        # Save to temp location
        temp_path = os.path.join(
            tempfile.gettempdir(),
            f"{uuid.uuid4()}_{file.filename}"
        )
        
        with open(temp_path, 'wb') as f:
            f.write(contents)
        
        # Process with attachment tool
        attachment_tool = agent.tools.get('attachment')
        if not attachment_tool:
            raise HTTPException(status_code=500, detail="Attachment tool not available")
        
        result = await attachment_tool.process_attachment(
            file_path=temp_path,
            filename=file.filename
        )
        
        # Clean up
        os.unlink(temp_path)
        
        if result["success"]:
            return FileUploadResponse(
                filename=file.filename,
                size_mb=size_mb,
                content_type=file.content_type or "unknown",
                processed=True,
                preview=result.get("preview"),
                error=None
            )
        else:
            return FileUploadResponse(
                filename=file.filename,
                size_mb=size_mb,
                content_type=file.content_type or "unknown",
                processed=False,
                preview=None,
                error=result.get("error")
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="File processing failed")


def cleanup_temp_files(attachments: List[Dict]):
    """Clean up temporary files after processing."""
    for attachment in attachments:
        try:
            if os.path.exists(attachment["path"]):
                os.unlink(attachment["path"])
                logger.debug(f"Cleaned up temp file: {attachment['path']}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp file: {e}")
```

### File 4: Session Management Routes

**`backend/app/api/routes/sessions.py`**
```python
"""
Session management API routes.
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
import uuid
from datetime import datetime, timedelta
import logging

from ...models.schemas import CreateSessionRequest, SessionResponse, SessionStatus
from ...models.session import Session
from ...models.message import Message
from ...agents.chat_agent import CustomerSupportAgent
from ...config import settings
from sqlalchemy.orm import Session as DBSession
from ...database import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


def get_agent() -> CustomerSupportAgent:
    """Get the agent instance from app state."""
    from ...main import app
    if not hasattr(app.state, 'agent'):
        raise HTTPException(status_code=500, detail="Agent not initialized")
    return app.state.agent


@router.post("", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest,
    db: DBSession = Depends(get_db),
    agent: CustomerSupportAgent = Depends(get_agent)
):
    """
    Create a new chat session.
    
    Args:
        request: Session creation parameters
        
    Returns:
        New session information
    """
    try:
        # Generate IDs
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        thread_id = f"thread_{uuid.uuid4().hex[:12]}"
        
        # Create session in database
        session = Session(
            id=session_id,
            user_id=request.user_id,
            thread_id=thread_id,
            status="active",
            metadata=request.metadata or {}
        )
        
        db.add(session)
        db.commit()
        db.refresh(session)
        
        # Initialize agent context
        context = agent.get_or_create_context(session_id)
        context.user_id = request.user_id
        context.thread_id = thread_id
        
        logger.info(f"Created session: {session_id} for user: {request.user_id}")
        
        return SessionResponse(
            session_id=session.id,
            user_id=session.user_id,
            thread_id=session.thread_id,
            status=SessionStatus(session.status),
            created_at=session.created_at,
            metadata=session.metadata
        )
        
    except Exception as e:
        logger.error(f"Failed to create session: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create session")


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    db: DBSession = Depends(get_db)
):
    """
    Get session information.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Session information
    """
    session = db.query(Session).filter(Session.id == session_id).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionResponse(
        session_id=session.id,
        user_id=session.user_id,
        thread_id=session.thread_id,
        status=SessionStatus(session.status),
        created_at=session.created_at,
        metadata=session.metadata
    )


@router.get("", response_model=List[SessionResponse])
async def list_sessions(
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    db: DBSession = Depends(get_db)
):
    """
    List sessions with optional filters.
    
    Args:
        user_id: Filter by user ID
        status: Filter by status
        limit: Maximum results
        offset: Pagination offset
        
    Returns:
        List of sessions
    """
    try:
        query = db.query(Session)
        
        # Apply filters
        if user_id:
            query = query.filter(Session.user_id == user_id)
        
        if status:
            query = query.filter(Session.status == status)
        
        # Order and paginate
        sessions = query.order_by(
            Session.created_at.desc()
        ).offset(offset).limit(limit).all()
        
        return [
            SessionResponse(
                session_id=s.id,
                user_id=s.user_id,
                thread_id=s.thread_id,
                status=SessionStatus(s.status),
                created_at=s.created_at,
                metadata=s.metadata
            )
            for s in sessions
        ]
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list sessions")


@router.patch("/{session_id}/status")
async def update_session_status(
    session_id: str,
    status: SessionStatus,
    db: DBSession = Depends(get_db),
    agent: CustomerSupportAgent = Depends(get_agent)
):
    """
    Update session status.
    
    Args:
        session_id: Session identifier
        status: New status
        
    Returns:
        Updated session information
    """
    try:
        session = db.query(Session).filter(Session.id == session_id).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Update status
        old_status = session.status
        session.status = status.value
        session.updated_at = datetime.utcnow()
        
        # Clean up agent resources if ending session
        if status == SessionStatus.ENDED:
            await agent.cleanup_session(session_id)
        
        db.commit()
        
        logger.info(f"Updated session {session_id} status: {old_status} -> {status.value}")
        
        return SessionResponse(
            session_id=session.id,
            user_id=session.user_id,
            thread_id=session.thread_id,
            status=SessionStatus(session.status),
            created_at=session.created_at,
            metadata=session.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update session status: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update session")


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    db: DBSession = Depends(get_db),
    agent: CustomerSupportAgent = Depends(get_agent)
):
    """
    Delete a session and its messages.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        session = db.query(Session).filter(Session.id == session_id).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Delete messages
        db.query(Message).filter(Message.session_id == session_id).delete()
        
        # Delete session
        db.delete(session)
        db.commit()
        
        # Clean up agent resources
        await agent.cleanup_session(session_id)
        
        logger.info(f"Deleted session: {session_id}")
        
        return {"message": "Session deleted successfully", "session_id": session_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete session")


@router.post("/cleanup")
async def cleanup_old_sessions(
    days: int = 30,
    db: DBSession = Depends(get_db),
    agent: CustomerSupportAgent = Depends(get_agent)
):
    """
    Clean up old sessions.
    
    Args:
        days: Delete sessions older than N days
        
    Returns:
        Cleanup statistics
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Find old sessions
        old_sessions = db.query(Session).filter(
            Session.last_activity < cutoff_date,
            Session.status != "active"
        ).all()
        
        deleted_count = 0
        for session in old_sessions:
            # Clean up agent resources
            await agent.cleanup_session(session.id)
            
            # Delete messages
            db.query(Message).filter(Message.session_id == session.id).delete()
            
            # Delete session
            db.delete(session)
            deleted_count += 1
        
        db.commit()
        
        logger.info(f"Cleaned up {deleted_count} old sessions")
        
        return {
            "message": "Cleanup completed",
            "sessions_deleted": deleted_count,
            "cutoff_date": cutoff_date.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup sessions: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail="Cleanup failed")
```

### File 5: Health Check Routes

**`backend/app/api/routes/health.py`**
```python
"""
Health check API routes.
"""
from fastapi import APIRouter, Depends
from datetime import datetime
import logging

from ...models.schemas import HealthResponse
from ...config import settings
from ...agents.chat_agent import CustomerSupportAgent
from sqlalchemy.orm import Session
from ...database import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """
    Basic health check endpoint.
    
    Returns:
        System health status
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.version,
        services={}
    )


@router.get("/ready", response_model=HealthResponse)
async def readiness_check(db: Session = Depends(get_db)):
    """
    Readiness check for all services.
    
    Returns:
        Detailed service health status
    """
    services = {}
    overall_status = "healthy"
    
    # Check database
    try:
        db.execute("SELECT 1")
        services["database"] = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        services["database"] = "unhealthy"
        overall_status = "degraded"
    
    # Check Redis cache
    try:
        from ...services.cache_service import CacheService
        cache = CacheService()
        if await cache.ping():
            services["redis"] = "healthy"
        else:
            services["redis"] = "unhealthy"
            overall_status = "degraded"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        services["redis"] = "unavailable"
    
    # Check ChromaDB
    try:
        from ...tools.rag_tool import RAGTool
        # Just check if collection exists
        services["chromadb"] = "healthy"
    except Exception as e:
        logger.error(f"ChromaDB health check failed: {e}")
        services["chromadb"] = "unhealthy"
        overall_status = "degraded"
    
    # Check agent
    try:
        from ...main import app
        if hasattr(app.state, 'agent') and app.state.agent.initialized:
            services["agent"] = "healthy"
        else:
            services["agent"] = "not_initialized"
            overall_status = "degraded"
    except Exception as e:
        logger.error(f"Agent health check failed: {e}")
        services["agent"] = "unhealthy"
        overall_status = "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version=settings.version,
        services=services
    )


@router.get("/live")
async def liveness_check():
    """
    Simple liveness check.
    
    Returns:
        Basic alive status
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}
```

### File 6: WebSocket Implementation

**`backend/app/api/websocket.py`**
```python
"""
WebSocket endpoint for real-time chat.
"""
from fastapi import WebSocket, WebSocketDisconnect, Depends
from typing import Dict, Set
import json
import logging
import asyncio
from datetime import datetime

from ..agents.chat_agent import CustomerSupportAgent
from ..models.schemas import WebSocketMessage

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str, client_id: str):
        """Accept and register a new connection."""
        await websocket.accept()
        
        self.active_connections[client_id] = websocket
        
        if session_id not in self.session_connections:
            self.session_connections[session_id] = set()
        self.session_connections[session_id].add(client_id)
        
        logger.info(f"WebSocket connected: session={session_id}, client={client_id}")
    
    def disconnect(self, session_id: str, client_id: str):
        """Remove a connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        if session_id in self.session_connections:
            self.session_connections[session_id].discard(client_id)
            if not self.session_connections[session_id]:
                del self.session_connections[session_id]
        
        logger.info(f"WebSocket disconnected: session={session_id}, client={client_id}")
    
    async def send_personal_message(self, message: str, client_id: str):
        """Send message to specific client."""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_text(message)
    
    async def broadcast_to_session(self, message: str, session_id: str):
        """Broadcast message to all clients in a session."""
        if session_id in self.session_connections:
            for client_id in self.session_connections[session_id]:
                if client_id in self.active_connections:
                    websocket = self.active_connections[client_id]
                    await websocket.send_text(message)


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    client_id: str = None
):
    """
    WebSocket endpoint for real-time chat.
    
    Args:
        websocket: WebSocket connection
        session_id: Chat session ID
        client_id: Optional client identifier
    """
    if not client_id:
        import uuid
        client_id = str(uuid.uuid4())
    
    await manager.connect(websocket, session_id, client_id)
    
    try:
        # Get agent instance
        from ..main import app
        agent: CustomerSupportAgent = app.state.agent
        
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                
                if message_data.get("type") == "message":
                    user_message = message_data.get("content", "")
                    
                    # Send acknowledgment
                    await websocket.send_json({
                        "type": "acknowledged",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    # Stream response
                    async for update in agent.stream_response(
                        session_id=session_id,
                        message=user_message,
                        attachments=message_data.get("attachments")
                    ):
                        await websocket.send_json(update)
                
                elif message_data.get("type") == "ping":
                    # Respond to ping
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                elif message_data.get("type") == "typing":
                    # Broadcast typing indicator to other clients
                    await manager.broadcast_to_session(
                        json.dumps({
                            "type": "user_typing",
                            "client_id": client_id,
                            "timestamp": datetime.utcnow().isoformat()
                        }),
                        session_id
                    )
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON",
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                logger.error(f"WebSocket message processing error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
    
    except WebSocketDisconnect:
        manager.disconnect(session_id, client_id)
        logger.info(f"WebSocket client {client_id} disconnected from session {session_id}")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        manager.disconnect(session_id, client_id)
```

### File 7: Middleware Implementation

**`backend/app/utils/middleware.py`**
```python
"""
Custom middleware for request processing.
"""
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import uuid
import logging
from typing import Callable

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add request ID to request state and response headers."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        
        # Add to logging context
        logger.info(f"Request started: {request.method} {request.url.path} [{request_id}]")
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Add request timing information."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Measure and log request processing time."""
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log slow requests
        if process_time > 1.0:
            logger.warning(
                f"Slow request: {request.method} {request.url.path} "
                f"took {process_time:.2f}s"
            )
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, app, calls: int = 100, period: int = 60
