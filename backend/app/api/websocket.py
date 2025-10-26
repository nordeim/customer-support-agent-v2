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
