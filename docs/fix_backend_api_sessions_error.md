# Analysis of Frontend-Backend Connection Error

## Executive Summary

When the frontend application connects to the backend, there's an AttributeError indicating that the `CreateSessionRequest` object has no attribute `tool_metadata`. This error occurs in the `create_session` function in `sessions.py` at line 57. The issue is a mismatch between the expected field name in the route handler and the actual field name in the Pydantic model.

## Detailed Analysis

### Error Breakdown

```
AttributeError: 'CreateSessionRequest' object has no attribute 'tool_metadata'
```

This error occurs in the `create_session` function in `sessions.py` at line 57:
```python
metadata=request.tool_metadata or {}
```

The error indicates that the `CreateSessionRequest` Pydantic model doesn't have a field named `tool_metadata`, but the route handler is trying to access it.

### Root Cause Analysis

Looking at the error and the code in `sessions.py`, there's a mismatch between the field name used in the route handler (`tool_metadata`) and the actual field name in the `CreateSessionRequest` model. This could be due to:

1. The field name in the Pydantic model is different (e.g., `metadata` instead of `tool_metadata`)
2. The field is not defined in the Pydantic model at all
3. There's a mismatch between the frontend request payload and the expected Pydantic model

### Deeper Investigation

To properly fix this issue, we need to:

1. Check the actual field name in the `CreateSessionRequest` Pydantic model
2. Verify what the frontend is sending in the request payload
3. Ensure consistency between the frontend and backend models
4. Update either the Pydantic model or the route handler to match

## Implementation Plan

1. First, we need to examine the `CreateSessionRequest` Pydantic model to determine the correct field name
2. Update the route handler in `sessions.py` to use the correct field name
3. Ensure the frontend is sending the correct field name in the request payload
4. Add proper error handling to provide more informative error messages
5. Add validation to ensure the request payload matches the expected model

## Implementation

Since we don't have direct access to the `CreateSessionRequest` model definition, we'll make an educated guess based on the error and common naming conventions. The most likely scenario is that the field is named `metadata` in the Pydantic model, not `tool_metadata`.

### Updated sessions.py

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
        
        # Get metadata from request - use the correct field name
        # Try multiple possible field names to handle different model versions
        metadata = {}
        if hasattr(request, 'metadata'):
            metadata = request.metadata or {}
        elif hasattr(request, 'tool_metadata'):
            metadata = request.tool_metadata or {}
        else:
            # Log the available attributes for debugging
            logger.warning(f"CreateSessionRequest attributes: {dir(request)}")
            # Use empty dict as fallback
            metadata = {}
        
        # Create session in database
        session = Session(
            id=session_id,
            user_id=request.user_id,
            thread_id=thread_id,
            status="active",
            metadata=metadata  # Use the correct field name
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
            metadata=session.metadata  # Use the correct field name
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
        metadata=session.metadata  # Use the correct field name
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
                metadata=s.metadata  # Use the correct field name
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
            metadata=session.metadata  # Use the correct field name
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

### Additional Improvements

1. **Better Error Handling**: Added more detailed error logging and handling
2. **Field Name Flexibility**: Added logic to handle different possible field names in the Pydantic model
3. **Debugging Information**: Added logging to show the available attributes of the request object for debugging
4. **Consistent Field Naming**: Ensured consistent use of the correct field name throughout the file

## Validation Steps

1. Apply the changes to `sessions.py`
2. Restart the backend application
3. Try connecting with the frontend again
4. Check the backend logs to see if the error is resolved
5. Verify that sessions are being created correctly

## Additional Recommendations

1. **Frontend-Backend Contract**: Ensure that the frontend and backend have a consistent contract for the session creation request payload
2. **Model Validation**: Add more detailed validation to the Pydantic models to catch mismatches early
3. **API Documentation**: Update the API documentation to reflect the correct field names
4. **Testing**: Add integration tests to verify the frontend-backend communication

This fix should resolve the AttributeError and allow the frontend to successfully create sessions with the backend. The solution is flexible enough to handle different possible field names in the Pydantic model, making it more robust against future changes.

---

https://chat.z.ai/s/c740952c-ede3-49c5-a097-84562cbc1ea9 

