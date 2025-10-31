"""
Session data validation using Pydantic.
Ensures session data integrity and type safety.

Version: 1.0.0
"""
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class SessionData(BaseModel):
    """
    Validated session data structure.
    
    Features:
    - Type validation
    - Constraint validation
    - JSON serialization validation
    - Fingerprint security validation
    """
    
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique session identifier"
    )
    
    user_id: Optional[str] = Field(
        None,
        max_length=255,
        description="User identifier"
    )
    
    thread_id: Optional[str] = Field(
        None,
        max_length=255,
        description="Thread identifier for conversation context"
    )
    
    message_count: int = Field(
        default=0,
        ge=0,
        le=100000,
        description="Number of messages in session"
    )
    
    escalated: bool = Field(
        default=False,
        description="Whether session has been escalated"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional session metadata"
    )
    
    request_id: Optional[str] = Field(
        None,
        max_length=255,
        description="Current request correlation ID"
    )
    
    created_at: Optional[datetime] = Field(
        None,
        description="Session creation timestamp"
    )
    
    updated_at: Optional[datetime] = Field(
        None,
        description="Last update timestamp"
    )
    
    last_activity: Optional[datetime] = Field(
        None,
        description="Last activity timestamp"
    )
    
    fingerprint: Optional[str] = Field(
        None,
        max_length=64,
        description="Session fingerprint for security"
    )
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        use_enum_values = True
        validate_assignment = True
    
    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure metadata is JSON-serializable.
        
        Args:
            v: Metadata dictionary
            
        Returns:
            Validated metadata
            
        Raises:
            ValueError: If metadata is not JSON-serializable
        """
        try:
            # Test JSON serialization
            json_str = json.dumps(v)
            
            # Check size
            if len(json_str) > 1_000_000:  # 1MB limit
                raise ValueError("Metadata exceeds maximum size (1MB)")
            
            return v
            
        except (TypeError, ValueError) as e:
            raise ValueError(f"Metadata must be JSON-serializable: {e}")
    
    @field_validator('message_count')
    @classmethod
    def validate_message_count(cls, v: int) -> int:
        """
        Validate message count is reasonable.
        
        Args:
            v: Message count
            
        Returns:
            Validated message count
        """
        if v < 0:
            raise ValueError("Message count cannot be negative")
        
        if v > 100000:
            raise ValueError("Message count exceeds maximum (100,000)")
        
        return v
    
    @field_validator('session_id', 'user_id', 'thread_id', 'request_id')
    @classmethod
    def validate_identifiers(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate identifier fields.
        
        Args:
            v: Identifier value
            
        Returns:
            Validated identifier
        """
        if v is None:
            return v
        
        # Check for SQL injection attempts
        dangerous_chars = ["'", '"', ';', '--', '/*', '*/', 'DROP', 'DELETE']
        v_upper = v.upper()
        
        for char in dangerous_chars:
            if char in v_upper:
                logger.warning(f"Potentially dangerous characters in identifier: {v}")
                raise ValueError("Invalid characters in identifier")
        
        # Check length
        if len(v) > 255:
            raise ValueError("Identifier too long (max 255 characters)")
        
        return v
    
    @field_validator('fingerprint')
    @classmethod
    def validate_fingerprint(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate session fingerprint format.
        
        Args:
            v: Fingerprint hash
            
        Returns:
            Validated fingerprint
        """
        if v is None:
            return v
        
        # Should be a hex string (SHA-256 = 64 chars)
        if not all(c in '0123456789abcdef' for c in v.lower()):
            raise ValueError("Fingerprint must be hexadecimal")
        
        if len(v) not in [32, 40, 64]:  # MD5, SHA-1, SHA-256
            raise ValueError("Invalid fingerprint length")
        
        return v.lower()
    
    @model_validator(mode='after')
    def validate_timestamps(self) -> 'SessionData':
        """
        Validate timestamp consistency.
        
        Returns:
            Validated session data
        """
        # Ensure created_at <= updated_at <= last_activity
        if self.created_at and self.updated_at:
            if self.updated_at < self.created_at:
                raise ValueError("updated_at cannot be before created_at")
        
        if self.updated_at and self.last_activity:
            if self.last_activity < self.updated_at:
                # This is acceptable - last_activity might be updated separately
                pass
        
        # Auto-set timestamps if missing
        now = datetime.utcnow()
        
        if not self.created_at:
            self.created_at = now
        
        if not self.updated_at:
            self.updated_at = now
        
        if not self.last_activity:
            self.last_activity = now
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        data = self.model_dump()
        
        # Convert datetime objects to ISO format strings
        for key in ['created_at', 'updated_at', 'last_activity']:
            if data.get(key) and isinstance(data[key], datetime):
                data[key] = data[key].isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionData':
        """
        Create SessionData from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            SessionData instance
        """
        # Make a copy to avoid modifying input
        data = dict(data)
        
        # Convert ISO format strings to datetime
        for key in ['created_at', 'updated_at', 'last_activity']:
            if key in data and data[key] is not None:
                if isinstance(data[key], str):
                    try:
                        data[key] = datetime.fromisoformat(data[key])
                    except ValueError as e:
                        logger.warning(f"Invalid datetime string for {key}: {data[key]}")
                        data[key] = None
                elif not isinstance(data[key], datetime):
                    logger.warning(f"Unexpected type for {key}: {type(data[key])}")
                    data[key] = None
        
        return cls(**data)
    
    def to_json(self) -> str:
        """
        Serialize to JSON string.
        
        Returns:
            JSON string
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SessionData':
        """
        Deserialize from JSON string.
        
        Args:
            json_str: JSON string
            
        Returns:
            SessionData instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def update_activity(self) -> None:
        """Update last_activity and updated_at timestamps."""
        now = datetime.utcnow()
        self.last_activity = now
        self.updated_at = now
    
    def increment_message_count(self, delta: int = 1) -> int:
        """
        Increment message count.
        
        Args:
            delta: Amount to increment by
            
        Returns:
            New message count
        """
        self.message_count += delta
        self.update_activity()
        return self.message_count
    
    def create_fingerprint(self, ip_address: str, user_agent: str) -> str:
        """
        Create session fingerprint from IP and user agent.
        
        Args:
            ip_address: Client IP address
            user_agent: Client user agent string
            
        Returns:
            Fingerprint hash
        """
        import hashlib
        
        data = f"{ip_address}:{user_agent}"
        fingerprint = hashlib.sha256(data.encode()).hexdigest()
        self.fingerprint = fingerprint
        
        return fingerprint
    
    def verify_fingerprint(self, ip_address: str, user_agent: str) -> bool:
        """
        Verify session fingerprint matches current request.
        
        Args:
            ip_address: Client IP address
            user_agent: Client user agent string
            
        Returns:
            True if fingerprint matches
        """
        if not self.fingerprint:
            # No fingerprint set (backward compatibility)
            return True
        
        import hashlib
        
        data = f"{ip_address}:{user_agent}"
        current_fingerprint = hashlib.sha256(data.encode()).hexdigest()
        
        return self.fingerprint == current_fingerprint


class SessionMetrics(BaseModel):
    """Session metrics for monitoring."""
    
    total_messages: int = Field(ge=0)
    avg_response_time: float = Field(ge=0.0)
    escalation_rate: float = Field(ge=0.0, le=1.0)
    active_sessions: int = Field(ge=0)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class SessionFilter(BaseModel):
    """Filter criteria for session queries."""
    
    user_id: Optional[str] = None
    escalated: Optional[bool] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    min_messages: Optional[int] = Field(None, ge=0)
    max_messages: Optional[int] = Field(None, ge=0)
    
    @model_validator(mode='after')
    def validate_date_range(self) -> 'SessionFilter':
        """Validate date range."""
        if self.created_after and self.created_before:
            if self.created_before < self.created_after:
                raise ValueError("created_before must be after created_after")
        
        return self
    
    @model_validator(mode='after')
    def validate_message_range(self) -> 'SessionFilter':
        """Validate message count range."""
        if self.min_messages is not None and self.max_messages is not None:
            if self.max_messages < self.min_messages:
                raise ValueError("max_messages must be >= min_messages")
        
        return self


__all__ = ['SessionData', 'SessionMetrics', 'SessionFilter']
