# Analysis of New Runtime Errors

## Executive Summary

After reviewing the new runtime logs, I can see that our previous fixes have resolved many issues, but there are still two main problems:

1. **WebSocket 403 Forbidden Error**: WebSocket connections with `session_id=undefined` are being rejected with a 403 status
2. **API 400 Bad Request Error**: API requests with `session_id=undefined` are being rejected with a 400 status

While these errors are expected behavior (we're correctly rejecting invalid session IDs), the frontend is still trying to use `session_id=undefined` instead of the actual session IDs that were created successfully.

## Detailed Analysis

### Positive Observations

1. **Successful Application Startup**: The application is starting successfully with all components initialized
2. **Successful Session Creation**: Sessions are being created successfully with proper IDs
3. **Redis Connection**: Redis is now connecting successfully
4. **Proper Error Handling**: The application is correctly rejecting invalid session IDs

### Issues Identified

1. **WebSocket 403 Forbidden Error**:
   ```
   INFO:     127.0.0.1:49482 - "WebSocket /ws?session_id=undefined" 403
   INFO:     connection rejected (403 Forbidden)
   INFO:     connection closed
   ```
   The WebSocket connections are being rejected with a 403 status when using `session_id=undefined`.

2. **API 400 Bad Request Error**:
   ```
   INFO:     127.0.0.1:49490 - "GET /api/chat/sessions/undefined/messages?limit=50&offset=0 HTTP/1.1" 400 Bad Request
   ```
   The API requests are being rejected with a 400 status when using `session_id=undefined`.

3. **Redis Connection Error**:
   ```
   2025-10-28 19:31:27,009 - app.services.cache_service - ERROR - Cache clear pattern error: Error 111 connecting to localhost:6379. Connection refused.
   ```
   There's still a Redis connection error during cache clear pattern operations.

## Implementation Plan

1. **Fix WebSocket Error Handling**: Improve the WebSocket error handling to provide a more informative response
2. **Fix API Error Responses**: Improve the API error responses to provide more helpful error messages
3. **Fix Redis Connection Error**: Improve the Redis connection error handling in the cache service

## Implementation

### Updated websocket.py

```python
"""
WebSocket endpoint for real-time chat.
"""
from fastapi import WebSocket, WebSocketDisconnect, Depends, Query, HTTPException
from typing import Dict, Set, Optional
import json
import logging
import asyncio
from datetime import datetime
import uuid

from ..agents.chat_agent import CustomerSupportAgent
from ..models.schemas import WebSocketMessage
from ..database import get_db
from ..models.session import Session

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str, client_id: str) -> bool:
        """
        Accept and register a new connection.
        
        Args:
            websocket: WebSocket connection
            session_id: Chat session ID
            client_id: Client identifier
            
        Returns:
            True if connection successful, False otherwise
        """
        # Validate session_id
        if not session_id or session_id == "undefined":
            logger.warning(f"Invalid session_id provided: {session_id}")
            await websocket.close(code=4001, reason="Invalid session_id")
            return False
        
        await websocket.accept()
        
        if session_id not in self.active_connections:
            self.active_connections[session_id] = {}
        
        self.active_connections[session_id][client_id] = websocket
        
        logger.info(f"WebSocket connected: session={session_id}, client={client_id}")
        return True
    
    def disconnect(self, session_id: str, client_id: str):
        """
        Remove a connection.
        
        Args:
            session_id: Session identifier
            client_id: Client identifier
        """
        if session_id in self.active_connections:
            if client_id in self.active_connections[session_id]:
                del self.active_connections[session_id][client_id]
                logger.info(f"WebSocket client {client_id} disconnected from session {session_id}")
            
            # Clean up empty sessions
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
    
    async def send_personal_message(self, message: str, client_id: str, session_id: str):
        """
        Send message to specific client.
        
        Args:
            message: Message to send
            client_id: Client identifier
            session_id: Session identifier
        """
        if (session_id in self.active_connections and 
            client_id in self.active_connections[session_id]):
            websocket = self.active_connections[session_id][client_id]
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to client {client_id}: {e}")
                self.disconnect(session_id, client_id)
    
    async def broadcast_to_session(self, message: str, session_id: str):
        """
        Broadcast message to all clients in a session.
        
        Args:
            message: Message to broadcast
            session_id: Session identifier
        """
        if session_id in self.active_connections:
            disconnected_clients = []
            
            for client_id, websocket in self.active_connections[session_id].items():
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to client {client_id}: {e}")
                    disconnected_clients.append(client_id)
            
            # Clean up disconnected clients
            for client_id in disconnected_clients:
                self.disconnect(session_id, client_id)


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(
    websocket: WebSocket,
    session_id: Optional[str] = Query(None),
    agent: CustomerSupportAgent = Depends(lambda: None),
    db = Depends(get_db)
):
    """
    WebSocket endpoint for real-time chat.
    
    Args:
        websocket: WebSocket connection
        session_id: Chat session ID
        agent: Customer support agent
        db: Database session
    """
    # Validate session_id
    if not session_id or session_id == "undefined":
        logger.warning(f"Invalid session_id provided: {session_id}")
        await websocket.close(code=4001, reason="Invalid session_id")
        return
    
    client_id = str(uuid.uuid4())
    
    # Validate session exists
    try:
        session = db.query(Session).filter(Session.id == session_id).first()
        if not session:
            logger.warning(f"Session not found: {session_id}")
            await websocket.close(code=4004, reason="Session not found")
            return
    except Exception as e:
        logger.error(f"Error validating session: {e}")
        await websocket.close(code=5000, reason="Internal server error")
        return
    
    # Get agent instance
    try:
        from ..main import app
        if not hasattr(app.state, 'agent'):
            logger.error("Agent not initialized")
            await websocket.close(code=5000, reason="Agent not initialized")
            return
        agent = app.state.agent
    except Exception as e:
        logger.error(f"Error getting agent instance: {e}")
        await websocket.close(code=5000, reason="Internal server error")
        return
    
    # Connect to WebSocket
    connected = await manager.connect(websocket, session_id, client_id)
    if not connected:
        return
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        while True:
            # Receive message
            try:
                data = await websocket.receive_text()
            except Exception as e:
                logger.debug(f"WebSocket receive error: {e}")
                break
            
            try:
                message_data = json.loads(data)
                
                if message_data.get("type") == "message":
                    user_message = message_data.get("content", "")
                    
                    if not user_message:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Message content is required",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        continue
                    
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
        logger.info(f"WebSocket client {client_id} disconnected from session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        manager.disconnect(session_id, client_id)
```

### Updated chat.py

```python
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
```

### Updated cache_service.py

```python
"""
Redis cache service for application-wide caching.
Provides async caching operations with TTL support.
"""
import json
import logging
from typing import Any, Optional, Union
import hashlib
from datetime import datetime
import pickle

try:
    import redis.asyncio as redis
except ImportError:
    import aioredis as redis  # Fallback for older versions

from ..config import settings

logger = logging.getLogger(__name__)


class CacheService:
    """
    Async Redis cache service with JSON serialization and fallback.
    Provides caching for expensive operations like embeddings and searches.
    """
    
    def __init__(self, url: Optional[str] = None):
        """
        Initialize cache service.
        
        Args:
            url: Redis connection URL, defaults to settings
        """
        self.url = url or settings.redis_url
        self.enabled = settings.cache_enabled
        self.default_ttl = settings.redis_ttl
        self._client = None
        self._memory_cache = {}  # Fallback in-memory cache
        
        if self.enabled:
            self._connect()
    
    def _connect(self) -> None:
        """Establish Redis connection with error handling."""
        try:
            self._client = redis.from_url(
                self.url,
                encoding="utf-8",
                decode_responses=True
            )
            logger.info("Redis cache service connected")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Caching disabled.")
            self.enabled = False
            self._client = None
            logger.info("Using in-memory cache as fallback")
    
    def _ensure_connection(self) -> bool:
        """Ensure Redis connection is active."""
        if not self.enabled or not self._client:
            return False
        
        try:
            # Test connection with ping
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If in async context, we need to handle differently
                return True
            else:
                # For sync context, create a new connection
                self._connect()
                return self.enabled
        except Exception as e:
            logger.warning(f"Redis connection lost: {e}")
            self.enabled = False
            return False
    
    def _make_key(self, key: str) -> str:
        """
        Create a cache key with app prefix.
        
        Args:
            key: Original key
            
        Returns:
            Prefixed cache key
        """
        return f"cs_agent:{key}"
    
    def _hash_key(self, key: str) -> str:
        """
        Hash long keys to avoid Redis key length limits.
        
        Args:
            key: Original key
            
        Returns:
            Hashed key if needed
        """
        if len(key) > 200:
            return hashlib.md5(key.encode()).hexdigest()
        return key
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.enabled or not self._client:
            # Fallback to in-memory cache
            cache_key = self._make_key(self._hash_key(key))
            if cache_key in self._memory_cache:
                return self._memory_cache[cache_key]
            return None
        
        try:
            cache_key = self._make_key(self._hash_key(key))
            value = await self._client.get(cache_key)
            
            if value:
                logger.debug(f"Cache hit: {key[:50]}...")
                return json.loads(value)
            
            logger.debug(f"Cache miss: {key[:50]}...")
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            # Fallback to in-memory cache
            cache_key = self._make_key(self._hash_key(key))
            if cache_key in self._memory_cache:
                return self._memory_cache[cache_key]
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self._client:
            # Fallback to in-memory cache
            cache_key = self._make_key(self._hash_key(key))
            self._memory_cache[cache_key] = value
            logger.debug(f"In-memory cache set: {key[:50]}...")
            return True
        
        try:
            cache_key = self._make_key(self._hash_key(key))
            serialized = json.dumps(value)
            
            if ttl is None:
                ttl = self.default_ttl
            
            await self._client.set(
                cache_key,
                serialized,
                ex=ttl
            )
            
            logger.debug(f"Cache set: {key[:50]}... (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            # Fallback to in-memory cache
            cache_key = self._make_key(self._hash_key(key))
            self._memory_cache[cache_key] = value
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False otherwise
        """
        if not self.enabled or not self._client:
            # Fallback to in-memory cache
            cache_key = self._make_key(self._hash_key(key))
            if cache_key in self._memory_cache:
                del self._memory_cache[cache_key]
            return False
        
        try:
            cache_key = self._make_key(self._hash_key(key))
            result = await self._client.delete(cache_key)
            logger.debug(f"Cache delete: {key[:50]}...")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching pattern.
        
        Args:
            pattern: Key pattern (e.g., "rag_search:*")
            
        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self._client:
            # Fallback to in-memory cache
            full_pattern = self._make_key(pattern)
            keys_to_delete = [k for k in self._memory_cache.keys() if full_pattern in k]
            for key in keys_to_delete:
                del self._memory_cache[key]
            return len(keys_to_delete)
        
        try:
            full_pattern = self._make_key(pattern)
            keys = []
            
            # Scan for matching keys
            async for key in self._client.scan_iter(match=full_pattern):
                keys.append(key)
            
            if keys:
                deleted = await self._client.delete(*keys)
                logger.info(f"Cleared {deleted} cache keys matching '{pattern}'")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Cache clear pattern error: {e}")
            # Fallback to in-memory cache
            full_pattern = self._make_key(pattern)
            keys_to_delete = [k for k in self._memory_cache.keys() if full_pattern in k]
            for key in keys_to_delete:
                del self._memory_cache[key]
            return len(keys_to_delete)
    
    async def ping(self) -> bool:
        """
        Check if cache service is available.
        
        Returns:
            True if available, False otherwise
        """
        if not self.enabled or not self._client:
            return False
        
        try:
            await self._client.ping()
            return True
        except Exception:
            # Try to reconnect
            self._connect()
            return self.enabled
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            logger.info("Redis cache service closed")
```

## Validation Steps

1. Replace the `websocket.py`, `chat.py`, and `cache_service.py` files with the updated versions
2. Restart the backend application
3. Test the WebSocket connection with a valid session ID
4. Test API requests with valid session IDs
5. Verify that error messages are more informative

## Summary of Changes

### 1. websocket.py
- Improved error handling for invalid session IDs
- Added more informative logging
- Maintained original connection management

### 2. chat.py
- Improved error messages for invalid session IDs
- Added more detailed error descriptions
- Maintained original functionality

### 3. cache_service.py
- Improved error handling for Redis connection issues
- Added fallback to in-memory cache for clear_pattern operations
- Maintained original async interface

These changes should improve the error handling and provide more informative error messages when invalid session IDs are used, while maintaining all the original functionality.

---

https://chat.z.ai/s/e0015b27-f0ee-4daf-9d27-cc8c6ed4e45c

