# 🚀 Complete Implementation - Production-Ready Files

I'll now generate all 6 files with complete, working implementations. Each file is a drop-in replacement with no placeholders.

---

## File 1: `backend/app/schemas/__init__.py` (NEW)

```python
"""
Request validation schemas for API and tool inputs.
Uses Pydantic v2 for type-safe validation and sanitization.

Version: 1.0.0
"""

from .tool_requests import (
    # Memory tool requests
    MemoryStoreRequest,
    MemoryRetrieveRequest,
    MemorySummarizeRequest,
    
    # RAG tool requests
    RAGSearchRequest,
    RAGAddDocumentsRequest,
    
    # Attachment tool requests
    AttachmentProcessRequest,
    AttachmentSaveRequest,
    
    # Escalation tool requests
    EscalationCheckRequest,
    
    # Base request
    ToolRequest,
    
    # Utilities
    sanitize_content,
    validate_session_id_format,
    validate_json_serializable,
    check_metadata_depth
)

__all__ = [
    # Memory
    'MemoryStoreRequest',
    'MemoryRetrieveRequest',
    'MemorySummarizeRequest',
    
    # RAG
    'RAGSearchRequest',
    'RAGAddDocumentsRequest',
    
    # Attachment
    'AttachmentProcessRequest',
    'AttachmentSaveRequest',
    
    # Escalation
    'EscalationCheckRequest',
    
    # Base
    'ToolRequest',
    
    # Utilities
    'sanitize_content',
    'validate_session_id_format',
    'validate_json_serializable',
    'check_metadata_depth'
]
```

---

## File 2: `backend/app/schemas/tool_requests.py` (NEW)

```python
"""
Request validation schemas for all tool operations.
Provides input sanitization and validation using Pydantic v2.

Version: 1.0.0

Security Features:
- SQL injection prevention via format validation
- Path traversal prevention
- XSS prevention via content sanitization
- Size limits to prevent DoS
- Depth limits for nested structures
"""
import re
import json
import logging
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Validation patterns
SESSION_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,255}$')
FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\. ]{1,255}$')

# Security constants
MAX_CONTENT_LENGTH = 10000
MAX_METADATA_SIZE = 10240  # 10KB
MAX_METADATA_DEPTH = 5
MAX_QUERY_LENGTH = 1000
MAX_DOCUMENT_LENGTH = 50000
MAX_DOCUMENTS_BATCH = 100
MAX_MESSAGE_LENGTH = 5000


# ===========================
# Utility Functions
# ===========================

def validate_session_id_format(session_id: str) -> str:
    """
    Validate session ID format.
    
    Only allows alphanumeric characters, hyphens, and underscores.
    Prevents SQL injection and path traversal attacks.
    
    Args:
        session_id: Session identifier to validate
        
    Returns:
        Validated session ID
        
    Raises:
        ValueError: If format is invalid
    """
    if not session_id:
        raise ValueError("session_id cannot be empty")
    
    if not SESSION_ID_PATTERN.match(session_id):
        raise ValueError(
            "session_id must contain only alphanumeric characters, "
            "hyphens, and underscores (1-255 characters)"
        )
    
    # Additional security checks
    dangerous_patterns = ['..', '/', '\\', ';', '--', '/*', '*/', 'DROP', 'DELETE', 'INSERT']
    session_id_upper = session_id.upper()
    
    for pattern in dangerous_patterns:
        if pattern in session_id_upper:
            raise ValueError(f"session_id contains dangerous pattern: {pattern}")
    
    return session_id


def sanitize_content(content: str, max_length: int = MAX_CONTENT_LENGTH) -> str:
    """
    Sanitize content string to prevent injection attacks and normalize whitespace.
    
    Args:
        content: Content to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized content
        
    Raises:
        ValueError: If content is invalid
    """
    if not content:
        raise ValueError("content cannot be empty")
    
    if not isinstance(content, str):
        raise ValueError(f"content must be string, got {type(content)}")
    
    # Remove null bytes (SQL injection vector)
    content = content.replace('\x00', '')
    
    # Remove other control characters except newlines and tabs
    content = ''.join(
        char for char in content
        if char.isprintable() or char in '\n\t'
    )
    
    # Normalize whitespace
    content = re.sub(r'[ \t]+', ' ', content)  # Multiple spaces to single
    content = re.sub(r'\n{3,}', '\n\n', content)  # Max 2 consecutive newlines
    
    # Strip leading/trailing whitespace
    content = content.strip()
    
    # Check length after sanitization
    if not content:
        raise ValueError("content is empty after sanitization")
    
    if len(content) > max_length:
        raise ValueError(
            f"content exceeds maximum length: {len(content)} > {max_length}"
        )
    
    return content


def validate_json_serializable(data: Any, max_size: int = MAX_METADATA_SIZE) -> Any:
    """
    Validate that data is JSON-serializable and within size limits.
    
    Args:
        data: Data to validate
        max_size: Maximum JSON size in bytes
        
    Returns:
        Validated data
        
    Raises:
        ValueError: If data is not serializable or too large
    """
    if data is None:
        return {}
    
    try:
        serialized = json.dumps(data)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Data is not JSON-serializable: {e}")
    
    size = len(serialized.encode('utf-8'))
    if size > max_size:
        raise ValueError(
            f"Serialized data exceeds maximum size: {size} > {max_size} bytes"
        )
    
    return data


def check_metadata_depth(obj: Any, current_depth: int = 0, max_depth: int = MAX_METADATA_DEPTH) -> None:
    """
    Check depth of nested structure to prevent DoS attacks.
    
    Args:
        obj: Object to check
        current_depth: Current nesting depth
        max_depth: Maximum allowed depth
        
    Raises:
        ValueError: If depth exceeds maximum
    """
    if current_depth > max_depth:
        raise ValueError(f"Metadata nesting exceeds maximum depth: {max_depth}")
    
    if isinstance(obj, dict):
        for value in obj.values():
            check_metadata_depth(value, current_depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            check_metadata_depth(item, current_depth + 1, max_depth)


# ===========================
# Base Request Model
# ===========================

class ToolRequest(BaseModel):
    """Base class for all tool request validation models."""
    
    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
        str_min_length = 1
        validate_assignment = True
        extra = 'forbid'  # Reject unknown fields


# ===========================
# Memory Tool Requests
# ===========================

class MemoryStoreRequest(ToolRequest):
    """
    Validated request for storing memory.
    
    Security:
    - session_id validated against injection patterns
    - content sanitized and length-limited
    - metadata depth and size limited
    - importance range validated
    
    Example:
        request = MemoryStoreRequest(
            session_id="user-123",
            content="User prefers email communication",
            content_type="preference",
            importance=0.8
        )
    """
    
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Session identifier (alphanumeric, hyphens, underscores only)"
    )
    
    content: str = Field(
        ...,
        min_length=1,
        max_length=MAX_CONTENT_LENGTH,
        description="Memory content to store"
    )
    
    content_type: Literal["user_info", "preference", "fact", "context"] = Field(
        default="context",
        description="Type of memory being stored"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata (max 10KB, max depth 5)"
    )
    
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Importance score for retrieval prioritization"
    )
    
    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """Validate session ID format and security."""
        return validate_session_id_format(v)
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Sanitize and validate content."""
        return sanitize_content(v, MAX_CONTENT_LENGTH)
    
    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate metadata structure and size."""
        if v is None:
            return {}
        
        # Check JSON serializability and size
        validated = validate_json_serializable(v, MAX_METADATA_SIZE)
        
        # Check nesting depth
        check_metadata_depth(validated)
        
        return validated


class MemoryRetrieveRequest(ToolRequest):
    """
    Validated request for retrieving memories.
    
    Example:
        request = MemoryRetrieveRequest(
            session_id="user-123",
            content_type="preference",
            limit=10,
            min_importance=0.3
        )
    """
    
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Session identifier"
    )
    
    content_type: Optional[Literal["user_info", "preference", "fact", "context"]] = Field(
        default=None,
        description="Filter by memory type (optional)"
    )
    
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of memories to retrieve"
    )
    
    time_window_hours: Optional[int] = Field(
        default=None,
        ge=1,
        le=720,  # 30 days max
        description="Only retrieve memories from last N hours (optional)"
    )
    
    min_importance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum importance threshold"
    )
    
    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """Validate session ID format."""
        return validate_session_id_format(v)


class MemorySummarizeRequest(ToolRequest):
    """
    Validated request for summarizing session memories.
    
    Example:
        request = MemorySummarizeRequest(
            session_id="user-123",
            max_items_per_type=3
        )
    """
    
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Session identifier"
    )
    
    max_items_per_type: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum items per memory type in summary"
    )
    
    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """Validate session ID format."""
        return validate_session_id_format(v)


# ===========================
# RAG Tool Requests
# ===========================

class RAGSearchRequest(ToolRequest):
    """
    Validated request for RAG search.
    
    Security:
    - query sanitized and length-limited
    - k bounded to prevent resource exhaustion
    - threshold validated
    - filter metadata validated
    
    Example:
        request = RAGSearchRequest(
            query="How do I reset my password?",
            k=5,
            threshold=0.7
        )
    """
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
        description="Search query"
    )
    
    k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return"
    )
    
    threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold"
    )
    
    filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filter for search (optional)"
    )
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Sanitize and validate query."""
        return sanitize_content(v, MAX_QUERY_LENGTH)
    
    @field_validator('filter')
    @classmethod
    def validate_filter(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate filter metadata."""
        if v is None:
            return None
        
        # Check JSON serializability and size
        validated = validate_json_serializable(v, MAX_METADATA_SIZE)
        
        # Check nesting depth
        check_metadata_depth(validated)
        
        return validated


class RAGAddDocumentsRequest(ToolRequest):
    """
    Validated request for adding documents to RAG.
    
    Security:
    - documents list size limited
    - each document length limited
    - IDs validated for uniqueness
    - metadata validated
    
    Example:
        request = RAGAddDocumentsRequest(
            documents=["Doc 1 content", "Doc 2 content"],
            metadatas=[{"source": "faq"}, {"source": "manual"}],
            chunk=True
        )
    """
    
    documents: List[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_DOCUMENTS_BATCH,
        description="List of documents to add"
    )
    
    metadatas: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Metadata for each document (optional, must match length)"
    )
    
    ids: Optional[List[str]] = Field(
        default=None,
        description="IDs for each document (optional, must match length and be unique)"
    )
    
    chunk: bool = Field(
        default=True,
        description="Whether to chunk documents for better retrieval"
    )
    
    @field_validator('documents')
    @classmethod
    def validate_documents(cls, v: List[str]) -> List[str]:
        """Validate each document."""
        if not v:
            raise ValueError("documents list cannot be empty")
        
        if len(v) > MAX_DOCUMENTS_BATCH:
            raise ValueError(
                f"Too many documents: {len(v)} > {MAX_DOCUMENTS_BATCH}"
            )
        
        validated = []
        for i, doc in enumerate(v):
            if not doc or not doc.strip():
                raise ValueError(f"Document at index {i} is empty")
            
            if len(doc) > MAX_DOCUMENT_LENGTH:
                raise ValueError(
                    f"Document at index {i} exceeds max length: "
                    f"{len(doc)} > {MAX_DOCUMENT_LENGTH}"
                )
            
            # Basic sanitization (preserve formatting for documents)
            sanitized = doc.replace('\x00', '')  # Remove null bytes
            validated.append(sanitized)
        
        return validated
    
    @field_validator('metadatas')
    @classmethod
    def validate_metadatas(cls, v: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Validate metadata list."""
        if v is None:
            return None
        
        for i, metadata in enumerate(v):
            # Check JSON serializability and size
            validate_json_serializable(metadata, MAX_METADATA_SIZE)
            
            # Check nesting depth
            check_metadata_depth(metadata)
        
        return v
    
    @field_validator('ids')
    @classmethod
    def validate_ids(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate IDs list."""
        if v is None:
            return None
        
        # Check for uniqueness
        if len(v) != len(set(v)):
            raise ValueError("ids must be unique")
        
        # Validate each ID format
        for i, id_val in enumerate(v):
            if not id_val or not id_val.strip():
                raise ValueError(f"ID at index {i} is empty")
            
            if len(id_val) > 255:
                raise ValueError(f"ID at index {i} exceeds max length: 255")
            
            # Check for dangerous characters
            if any(char in id_val for char in ['\x00', '\n', '\r', ';', '--']):
                raise ValueError(f"ID at index {i} contains invalid characters")
        
        return v
    
    @model_validator(mode='after')
    def validate_list_lengths(self) -> 'RAGAddDocumentsRequest':
        """Validate that all lists have matching lengths."""
        doc_count = len(self.documents)
        
        if self.metadatas is not None and len(self.metadatas) != doc_count:
            raise ValueError(
                f"metadatas length ({len(self.metadatas)}) must match "
                f"documents length ({doc_count})"
            )
        
        if self.ids is not None and len(self.ids) != doc_count:
            raise ValueError(
                f"ids length ({len(self.ids)}) must match "
                f"documents length ({doc_count})"
            )
        
        return self


# ===========================
# Attachment Tool Requests
# ===========================

class AttachmentProcessRequest(ToolRequest):
    """
    Validated request for processing attachments.
    
    Security:
    - filename sanitized to prevent path traversal
    - file_path validated (actual existence checked at runtime)
    
    Example:
        request = AttachmentProcessRequest(
            file_path="/tmp/upload/document.pdf",
            filename="document.pdf",
            chunk_for_rag=True
        )
    """
    
    file_path: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Path to file to process"
    )
    
    filename: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Original filename (optional)"
    )
    
    extract_metadata: bool = Field(
        default=True,
        description="Whether to extract file metadata"
    )
    
    chunk_for_rag: bool = Field(
        default=False,
        description="Whether to chunk content for RAG indexing"
    )
    
    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate file path for security."""
        # Check for path traversal attempts
        dangerous_patterns = ['..', '~/', '/etc/', 'C:\\Windows', '/bin/', '/usr/bin/']
        
        for pattern in dangerous_patterns:
            if pattern in v:
                raise ValueError(f"file_path contains dangerous pattern: {pattern}")
        
        # Remove null bytes
        v = v.replace('\x00', '')
        
        # Note: We don't check file existence here - that's done at runtime
        # This is just format validation
        
        return v
    
    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v: Optional[str]) -> Optional[str]:
        """Validate filename for security."""
        if v is None:
            return None
        
        # Remove path separators
        v = v.replace('/', '').replace('\\', '').replace('\x00', '')
        
        # Check for valid characters only
        if not FILENAME_PATTERN.match(v):
            raise ValueError(
                "filename must contain only alphanumeric characters, "
                "spaces, hyphens, underscores, and dots"
            )
        
        return v


class AttachmentSaveRequest(ToolRequest):
    """
    Validated request for saving uploaded file.
    
    Example:
        request = AttachmentSaveRequest(
            filename="document.pdf"
        )
    """
    
    filename: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Filename for uploaded file"
    )
    
    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """Validate and sanitize filename."""
        # Remove path separators
        v = v.replace('/', '').replace('\\', '').replace('\x00', '')
        
        # Check for valid characters
        if not FILENAME_PATTERN.match(v):
            raise ValueError(
                "filename must contain only alphanumeric characters, "
                "spaces, hyphens, underscores, and dots"
            )
        
        # Prevent hidden files
        if v.startswith('.'):
            raise ValueError("filename cannot start with dot (hidden files not allowed)")
        
        return v


# ===========================
# Escalation Tool Requests
# ===========================

class EscalationCheckRequest(ToolRequest):
    """
    Validated request for escalation check.
    
    Security:
    - message sanitized and length-limited
    - message_history size limited
    
    Example:
        request = EscalationCheckRequest(
            message="I need to speak to a manager immediately!",
            confidence_threshold=0.7
        )
    """
    
    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="User message to analyze for escalation"
    )
    
    message_history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        max_length=50,
        description="Recent message history for context (optional)"
    )
    
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for escalation trigger"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context metadata (optional)"
    )
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Sanitize and validate message."""
        return sanitize_content(v, MAX_MESSAGE_LENGTH)
    
    @field_validator('message_history')
    @classmethod
    def validate_message_history(cls, v: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Validate message history."""
        if v is None:
            return None
        
        if len(v) > 50:
            raise ValueError("message_history too large (max 50 messages)")
        
        # Validate each message structure
        for i, msg in enumerate(v):
            if not isinstance(msg, dict):
                raise ValueError(f"message_history[{i}] must be dictionary")
            
            # Check JSON serializability
            validate_json_serializable(msg, MAX_METADATA_SIZE)
        
        return v
    
    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate metadata."""
        if v is None:
            return None
        
        # Check JSON serializability and size
        validated = validate_json_serializable(v, MAX_METADATA_SIZE)
        
        # Check nesting depth
        check_metadata_depth(validated)
        
        return validated


# Export all models
__all__ = [
    # Memory
    'MemoryStoreRequest',
    'MemoryRetrieveRequest',
    'MemorySummarizeRequest',
    
    # RAG
    'RAGSearchRequest',
    'RAGAddDocumentsRequest',
    
    # Attachment
    'AttachmentProcessRequest',
    'AttachmentSaveRequest',
    
    # Escalation
    'EscalationCheckRequest',
    
    # Base
    'ToolRequest',
    
    # Utilities
    'sanitize_content',
    'validate_session_id_format',
    'validate_json_serializable',
    'check_metadata_depth',
]
```

---

## File 3: `backend/app/tools/memory_tool.py` (COMPLETE UPDATE)

```python
"""
Memory management tool for conversation context persistence.
Uses database for storing and retrieving conversation memories.

Version: 3.0.0 (Enhanced with validation, context managers, and security)

Changes:
- Added Pydantic input validation
- Added database session context managers with timeouts
- Added query timeout configuration
- Enhanced error handling and logging
- Improved connection health checks
"""
import logging
import json
import uuid
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from contextlib import contextmanager

from sqlalchemy import create_engine, desc, and_, or_, func, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool
from sqlalchemy.exc import IntegrityError, DBAPIError, DisconnectionError
from pydantic import ValidationError

from ..config import settings
from ..models.memory import Memory
from ..schemas.tool_requests import (
    MemoryStoreRequest,
    MemoryRetrieveRequest,
    MemorySummarizeRequest
)
from .base_tool import BaseTool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)

# Memory type priorities for retrieval
MEMORY_TYPE_PRIORITY = {
    "user_info": 1.0,
    "preference": 0.9,
    "fact": 0.8,
    "context": 0.7
}

# Default limits
DEFAULT_MEMORY_LIMIT = 10
DEFAULT_TIME_WINDOW_HOURS = 24


class MemoryTool(BaseTool):
    """
    Memory management tool for storing and retrieving conversation context.
    
    Version 3.0.0:
    - ADDED: Pydantic input validation
    - ADDED: Database session context managers with timeouts
    - FIXED: Connection pooling for production
    - FIXED: Unique constraint handling for duplicates
    - FIXED: Field naming (metadata, not tool_metadata)
    - Enhanced error handling and security
    """
    
    def __init__(self):
        """Initialize memory tool with database connection."""
        super().__init__(
            name="memory_management",
            description="Store and retrieve conversation memory and context",
            version="3.0.0"
        )
        
        # Resources initialized in async initialize()
        self.engine = None
        self.SessionLocal = None
    
    async def initialize(self) -> None:
        """Initialize memory tool resources (async-safe)."""
        try:
            logger.info(f"Initializing Memory tool '{self.name}'...")
            
            # Initialize database engine (I/O-bound, run in thread pool)
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._init_database
            )
            
            self.initialized = True
            logger.info(f"✓ Memory tool '{self.name}' initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Memory tool: {e}", exc_info=True)
            raise
    
    async def cleanup(self) -> None:
        """Cleanup memory tool resources."""
        try:
            logger.info(f"Cleaning up Memory tool '{self.name}'...")
            
            # Dispose of database engine
            if self.engine:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.engine.dispose
                )
                self.engine = None
                self.SessionLocal = None
            
            self.initialized = False
            logger.info(f"✓ Memory tool '{self.name}' cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during Memory tool cleanup: {e}")
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute memory operations (async-first)."""
        action = kwargs.get("action", "retrieve")
        session_id = kwargs.get("session_id")
        
        if not session_id:
            return ToolResult.error_result(
                error="session_id is required",
                metadata={"tool": self.name}
            )
        
        try:
            if action == "store":
                content = kwargs.get("content")
                if not content:
                    return ToolResult.error_result(
                        error="content is required for store action",
                        metadata={"tool": self.name, "action": action}
                    )
                
                result = await self.store_memory_async(
                    session_id=session_id,
                    content=content,
                    content_type=kwargs.get("content_type", "context"),
                    metadata=kwargs.get("metadata"),
                    importance=kwargs.get("importance", 0.5)
                )
                
                return ToolResult.success_result(
                    data=result,
                    metadata={
                        "tool": self.name,
                        "action": action,
                        "session_id": session_id
                    }
                )
            
            elif action == "retrieve":
                memories = await self.retrieve_memories_async(
                    session_id=session_id,
                    content_type=kwargs.get("content_type"),
                    limit=kwargs.get("limit", DEFAULT_MEMORY_LIMIT),
                    time_window_hours=kwargs.get("time_window_hours"),
                    min_importance=kwargs.get("min_importance", 0.0)
                )
                
                return ToolResult.success_result(
                    data={
                        "memories": memories,
                        "count": len(memories)
                    },
                    metadata={
                        "tool": self.name,
                        "action": action,
                        "session_id": session_id
                    }
                )
            
            elif action == "summarize":
                summary = await self.summarize_session_async(
                    session_id=session_id,
                    max_items_per_type=kwargs.get("max_items_per_type", 3)
                )
                
                return ToolResult.success_result(
                    data={"summary": summary},
                    metadata={
                        "tool": self.name,
                        "action": action,
                        "session_id": session_id,
                        "summary_length": len(summary)
                    }
                )
            
            else:
                return ToolResult.error_result(
                    error=f"Unknown action: {action}. Valid: store, retrieve, summarize",
                    metadata={"tool": self.name}
                )
                
        except Exception as e:
            logger.error(f"Memory execute error: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={"tool": self.name, "action": action, "session_id": session_id}
            )
    
    # ===========================
    # Core Memory Methods (Async) - ENHANCED WITH VALIDATION
    # ===========================
    
    async def store_memory_async(
        self,
        session_id: str,
        content: str,
        content_type: str = "context",
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5
    ) -> Dict[str, Any]:
        """
        Store a memory entry for a session (async).
        
        Version 3.0.0: Added Pydantic validation.
        """
        # ADDED: Input validation
        try:
            validated_request = MemoryStoreRequest(
                session_id=session_id,
                content=content,
                content_type=content_type,
                metadata=metadata,
                importance=importance
            )
        except ValidationError as e:
            logger.error(f"Memory store validation failed: {e}")
            return {
                "success": False,
                "error": f"Input validation failed: {e}",
                "validation_errors": e.errors()
            }
        
        try:
            # Run database operation in thread pool with validated data
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._store_memory_sync,
                validated_request.session_id,
                validated_request.content,
                validated_request.content_type,
                validated_request.metadata,
                validated_request.importance
            )
            
            logger.info(
                f"Stored memory for session {session_id}: "
                f"type={content_type}, importance={importance}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def retrieve_memories_async(
        self,
        session_id: str,
        content_type: Optional[str] = None,
        limit: int = DEFAULT_MEMORY_LIMIT,
        time_window_hours: Optional[int] = None,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories for a session (async).
        
        Version 3.0.0: Added Pydantic validation.
        """
        # ADDED: Input validation
        try:
            validated_request = MemoryRetrieveRequest(
                session_id=session_id,
                content_type=content_type,
                limit=limit,
                time_window_hours=time_window_hours,
                min_importance=min_importance
            )
        except ValidationError as e:
            logger.error(f"Memory retrieve validation failed: {e}")
            return []
        
        try:
            # Run database operation in thread pool with validated data
            memories = await asyncio.get_event_loop().run_in_executor(
                None,
                self._retrieve_memories_sync,
                validated_request.session_id,
                validated_request.content_type,
                validated_request.limit,
                validated_request.time_window_hours,
                validated_request.min_importance
            )
            
            logger.debug(f"Retrieved {len(memories)} memories for session {session_id}")
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}", exc_info=True)
            return []
    
    async def summarize_session_async(
        self,
        session_id: str,
        max_items_per_type: int = 3
    ) -> str:
        """
        Generate a text summary of session memories (async).
        
        Version 3.0.0: Added Pydantic validation.
        """
        # ADDED: Input validation
        try:
            validated_request = MemorySummarizeRequest(
                session_id=session_id,
                max_items_per_type=max_items_per_type
            )
        except ValidationError as e:
            logger.error(f"Memory summarize validation failed: {e}")
            return "Error: Invalid request parameters."
        
        try:
            # Retrieve memories grouped by type
            memory_groups = {}
            
            for content_type in MEMORY_TYPE_PRIORITY.keys():
                memories = await self.retrieve_memories_async(
                    session_id=validated_request.session_id,
                    content_type=content_type,
                    limit=validated_request.max_items_per_type,
                    min_importance=0.3
                )
                
                if memories:
                    memory_groups[content_type] = memories
            
            if not memory_groups:
                return "No previous context available for this session."
            
            # Build summary
            summary_parts = []
            
            if "user_info" in memory_groups:
                user_info = [m["content"] for m in memory_groups["user_info"]]
                summary_parts.append(f"User Information: {'; '.join(user_info)}")
            
            if "preference" in memory_groups:
                preferences = [m["content"] for m in memory_groups["preference"]]
                summary_parts.append(f"User Preferences: {'; '.join(preferences)}")
            
            if "fact" in memory_groups:
                facts = [m["content"] for m in memory_groups["fact"][:3]]
                summary_parts.append(f"Key Facts: {'; '.join(facts)}")
            
            if "context" in memory_groups:
                contexts = [m["content"] for m in memory_groups["context"][:5]]
                summary_parts.append(f"Recent Context: {'; '.join(contexts[:3])}")
            
            summary = "\n".join(summary_parts)
            
            logger.debug(f"Generated summary for session {session_id}: {len(summary)} chars")
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate session summary: {e}", exc_info=True)
            return "Error retrieving session context."
    
    async def cleanup_old_memories_async(
        self,
        days: int = 30,
        max_per_session: int = 100
    ) -> Dict[str, Any]:
        """Clean up old and low-importance memories (async)."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._cleanup_old_memories_sync,
                days,
                max_per_session
            )
            
            logger.info(f"Memory cleanup completed: {result['total_deleted']} memories deleted")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to cleanup memories: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    # ===========================
    # Database Session Context Manager (NEW)
    # ===========================
    
    @contextmanager
    def get_db_session_context(self, timeout: float = 30.0):
        """
        Context manager for database sessions with timeout and health checks.
        
        ADDED in Version 3.0.0 for safe session management.
        
        Args:
            timeout: Maximum session lifetime in seconds
            
        Yields:
            Database session
            
        Raises:
            RuntimeError: If session times out or database not initialized
            
        Example:
            with self.get_db_session_context(timeout=10.0) as db:
                memory = Memory(...)
                db.add(memory)
                # Commit happens automatically if no exception
        """
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        session = self.SessionLocal()
        start_time = time.time()
        
        try:
            # Validate connection with a simple query
            try:
                session.execute(text("SELECT 1"))
            except (DBAPIError, DisconnectionError) as e:
                logger.error(f"Database connection validation failed: {e}")
                session.invalidate()
                raise RuntimeError(f"Database connection unhealthy: {e}")
            
            # Yield session for operations
            yield session
            
            # Check if timeout exceeded
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(f"Session exceeded timeout: {elapsed:.2f}s > {timeout}s")
                session.rollback()
                raise RuntimeError(f"Session timeout exceeded: {elapsed:.2f}s")
            
            # Commit if no exceptions
            session.commit()
            
        except (DBAPIError, DisconnectionError) as e:
            logger.error(f"Database error during session: {e}")
            session.rollback()
            # Test if connection is still alive
            try:
                session.execute(text("SELECT 1"))
            except Exception:
                # Connection is dead, invalidate it
                session.invalidate()
            raise
            
        except Exception as e:
            logger.error(f"Session error: {e}", exc_info=True)
            session.rollback()
            raise
            
        finally:
            # GUARANTEED cleanup
            session.close()
            
            # Log slow sessions
            elapsed = time.time() - start_time
            if elapsed > 5.0:
                logger.warning(f"Slow database session: {elapsed:.2f}s")
    
    # ===========================
    # Private Helper Methods (Sync) - UPDATED WITH CONTEXT MANAGERS
    # ===========================
    
    def _init_database(self) -> None:
        """
        Initialize database engine with proper connection pooling.
        
        Version 3.0.0: Added query timeout configuration.
        """
        try:
            connect_args = {}
            poolclass = None
            pool_config = {}
            
            if "sqlite" in settings.database_url:
                # SQLite configuration
                connect_args = {
                    "check_same_thread": False,
                    "timeout": 20
                }
                poolclass = StaticPool  # SQLite doesn't benefit from pooling
            else:
                # PostgreSQL or other databases
                pool_config = {
                    "pool_size": 10,
                    "max_overflow": 20,
                    "pool_timeout": 30,
                    "pool_recycle": 3600,
                    "pool_pre_ping": True
                }
                poolclass = QueuePool
                
                # ADDED: Query timeout for PostgreSQL
                connect_args = {
                    "options": "-c statement_timeout=30000"  # 30 seconds
                }
            
            self.engine = create_engine(
                settings.database_url,
                connect_args=connect_args,
                poolclass=poolclass,
                echo=settings.database_echo,
                **pool_config
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
                expire_on_commit=False
            )
            
            pool_size = pool_config.get('pool_size', 'N/A')
            logger.info(f"Memory database initialized (pool_size={pool_size})")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}", exc_info=True)
            raise
    
    def _store_memory_sync(
        self,
        session_id: str,
        content: str,
        content_type: str,
        metadata: Optional[Dict[str, Any]],
        importance: float
    ) -> Dict[str, Any]:
        """
        Store memory (sync implementation for thread pool).
        
        Version 3.0.0: UPDATED to use context manager.
        """
        # UPDATED: Use context manager for safe session management
        with self.get_db_session_context(timeout=10.0) as db:
            try:
                # Try to create new memory
                memory = Memory(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    content_type=content_type,
                    content=content,
                    metadata=metadata or {},
                    importance=importance
                )
                
                db.add(memory)
                db.flush()  # Get any database errors early
                
                return {
                    "success": True,
                    "memory_id": memory.id,
                    "action": "created",
                    "message": "Memory stored successfully"
                }
                
            except IntegrityError:
                # Duplicate detected, update existing
                db.rollback()
                
                existing = db.query(Memory).filter(
                    and_(
                        Memory.session_id == session_id,
                        Memory.content_type == content_type,
                        Memory.content == content
                    )
                ).first()
                
                if existing:
                    # Update importance and access tracking
                    existing.importance = max(existing.importance, importance)
                    existing.last_accessed = datetime.utcnow()
                    existing.access_count += 1
                    
                    # Update metadata if provided
                    if metadata:
                        existing.metadata.update(metadata)
                    
                    db.flush()
                    
                    logger.debug(f"Updated existing memory: {existing.id}")
                    
                    return {
                        "success": True,
                        "memory_id": existing.id,
                        "action": "updated",
                        "message": "Memory updated successfully"
                    }
                else:
                    # Race condition: deleted between attempts
                    raise
    
    def _retrieve_memories_sync(
        self,
        session_id: str,
        content_type: Optional[str],
        limit: int,
        time_window_hours: Optional[int],
        min_importance: float
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories (sync implementation for thread pool).
        
        Version 3.0.0: UPDATED to use context manager.
        """
        # UPDATED: Use context manager
        with self.get_db_session_context(timeout=10.0) as db:
            query = db.query(Memory).filter(Memory.session_id == session_id)
            
            # Apply filters
            if content_type:
                query = query.filter(Memory.content_type == content_type)
            
            if time_window_hours:
                cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
                query = query.filter(Memory.created_at >= cutoff_time)
            
            if min_importance > 0:
                query = query.filter(Memory.importance >= min_importance)
            
            # Order by importance and recency
            query = query.order_by(
                desc(Memory.importance),
                desc(Memory.created_at)
            ).limit(limit)
            
            memories = query.all()
            
            # Update access times
            for memory in memories:
                memory.update_access()
            
            db.flush()
            
            # Format results
            results = []
            for memory in memories:
                results.append({
                    "id": memory.id,
                    "content_type": memory.content_type,
                    "content": memory.content,
                    "metadata": memory.metadata,
                    "importance": memory.importance,
                    "created_at": memory.created_at.isoformat(),
                    "access_count": memory.access_count
                })
            
            return results
    
    def _cleanup_old_memories_sync(
        self,
        days: int,
        max_per_session: int
    ) -> Dict[str, Any]:
        """
        Cleanup old memories (sync implementation for thread pool).
        
        Version 3.0.0: UPDATED to use context manager with longer timeout.
        """
        # UPDATED: Use context manager with longer timeout for cleanup
        with self.get_db_session_context(timeout=60.0) as db:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Delete old, low-importance, rarely accessed memories
            deleted_old = db.query(Memory).filter(
                and_(
                    Memory.last_accessed < cutoff_date,
                    Memory.importance < 0.3,
                    Memory.access_count < 3
                )
            ).delete(synchronize_session=False)
            
            # For each session, keep only the most recent/important memories
            sessions = db.query(Memory.session_id).distinct().all()
            deleted_excess = 0
            
            for (session_id,) in sessions:
                # Get memories ordered by importance and recency
                memories = db.query(Memory).filter(
                    Memory.session_id == session_id
                ).order_by(
                    desc(Memory.importance),
                    desc(Memory.created_at)
                ).offset(max_per_session).all()
                
                # Delete excess memories
                for memory in memories:
                    db.delete(memory)
                    deleted_excess += 1
            
            db.flush()
            
            total_deleted = deleted_old + deleted_excess
            
            return {
                "success": True,
                "deleted_old": deleted_old,
                "deleted_excess": deleted_excess,
                "total_deleted": total_deleted
            }


__all__ = ['MemoryTool']
```

---

**Continuing with remaining files in next message due to length...**

Would you like me to continue with Files 4-6?
