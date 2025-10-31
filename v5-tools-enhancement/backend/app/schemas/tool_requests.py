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
