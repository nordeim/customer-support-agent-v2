"""
WebSocket endpoint for real-time chat.
"""
from fastapi import WebSocket, WebSocketDisconnect, Depends, Query
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
    if not session_id:
        await websocket.close(code=4001, reason="Session ID is required")
        return
    
    client_id = str(uuid.uuid4())
    
    # Validate session exists
    try:
        session = db.query(Session).filter(Session.id == session_id).first()
        if not session:
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
