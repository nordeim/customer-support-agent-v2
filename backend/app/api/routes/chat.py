"""
Chat API routes for message handling.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict
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
    # Validate session_id
    if not session_id or session_id == "undefined":
        raise HTTPException(
            status_code=400, 
            detail="Invalid session_id. Please create a new session or provide a valid session ID."
        )
    
    try:
        # Validate session exists
        session = db.query(Session).filter(Session.id == session_id).first()
        if not session:
            raise HTTPException(
                status_code=404, 
                detail=f"Session not found with ID: {session_id}. Please create a new session."
            )
        
        # Check session status
        if session.status != "active":
            raise HTTPException(
                status_code=400,
                detail=f"Session is {session.status}. Cannot send messages to inactive sessions."
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
            if agent_response.tool_metadata.get("ticket_id"):
                session.escalation_ticket_id = agent_response.tool_metadata["ticket_id"]
        
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
            metadata=agent_response.tool_metadata
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
    # Validate session_id
    if not session_id or session_id == "undefined":
        raise HTTPException(
            status_code=400, 
            detail="Invalid session_id. Please provide a valid session ID."
        )
    
    try:
        # Validate session exists
        session = db.query(Session).filter(Session.id == session_id).first()
        if not session:
            raise HTTPException(
                status_code=404, 
                detail=f"Session not found with ID: {session_id}. Please create a new session."
            )
        
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
    # Validate session_id
    if not session_id or session_id == "undefined":
        raise HTTPException(
            status_code=400, 
            detail="Invalid session_id. Please provide a valid session ID."
        )
    
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
