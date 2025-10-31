"""
Memory model for conversation context persistence.
Stores important information extracted from conversations.

Version: 3.0.0 (Enhanced with content hashing, TTL, and soft delete)

Changes from 2.0.0:
- Added content_hash for duplicate detection
- Added semantic_hash for similarity detection
- Added expires_at for automatic TTL
- Added deleted_at for soft delete (GDPR compliance)
- Updated unique constraint to use content_hash
- Added composite indexes with partial conditions
- Added importance decay calculation
- Enhanced validation and helper methods
"""
from sqlalchemy import (
    Column, String, Text, Float, Integer, DateTime, JSON,
    Index, UniqueConstraint, CheckConstraint, text
)
from sqlalchemy.sql import func
from sqlalchemy.ext.hybrid import hybrid_property
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json
import hashlib
import math
import uuid

from ..database import Base


class Memory(Base):
    """
    Memory storage for conversation context with content hashing and TTL.
    
    Stores important facts, preferences, and context extracted
    from user conversations for enhanced personalization.
    
    Version 3.0.0:
    - Content hash for exact duplicate detection
    - Semantic hash for similarity clustering
    - TTL support for automatic cleanup
    - Soft delete for GDPR compliance
    - Importance decay over time
    - Optimized indexes for performance
    """
    
    __tablename__ = "memories"
    
    # Primary key
    id = Column(String(36), primary_key=True, index=True)
    
    # Session reference
    session_id = Column(String(255), nullable=False, index=True)
    
    # Memory classification
    content_type = Column(
        String(50),
        nullable=False,
        default="context",
        comment="Type: user_info, preference, fact, context"
    )
    
    # Memory content
    content = Column(Text, nullable=False)
    
    # ADDED: Content hash for duplicate detection
    content_hash = Column(
        String(64),
        nullable=False,
        index=True,
        comment="SHA256 hash of normalized content for exact duplicate detection"
    )
    
    # ADDED: Semantic hash for similarity clustering
    semantic_hash = Column(
        String(64),
        nullable=True,
        index=True,
        comment="SimHash for semantic similarity detection (optional)"
    )
    
    # Metadata
    metadata = Column(
        JSON,
        nullable=False,
        default=dict,
        comment="Additional metadata about the memory"
    )
    
    # Importance score (0.0 to 1.0)
    importance = Column(
        Float,
        nullable=False,
        default=0.5,
        comment="Importance score for retrieval prioritization"
    )
    
    # ADDED: TTL support
    expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Expiration timestamp for automatic cleanup"
    )
    
    # Access tracking
    access_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of times this memory was accessed"
    )
    
    last_accessed = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last time this memory was retrieved"
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # ADDED: Soft delete support
    deleted_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Soft delete timestamp for GDPR compliance"
    )
    
    # Table-level constraints and indexes
    __table_args__ = (
        # UPDATED: Unique constraint on content hash instead of full content
        UniqueConstraint(
            'session_id',
            'content_type',
            'content_hash',
            name='uq_memory_session_content_hash'
        ),
        
        # ADDED: Composite index for efficient retrieval (with partial condition)
        Index(
            'ix_memory_session_type_importance',
            'session_id',
            'content_type',
            'importance',
            postgresql_where=text('deleted_at IS NULL')  # PostgreSQL partial index
        ),
        
        # ADDED: Index for active memories with expiration
        Index(
            'ix_memory_active_expires',
            'expires_at',
            postgresql_where=text('deleted_at IS NULL AND expires_at IS NOT NULL')
        ),
        
        # ADDED: Composite index for cleanup queries
        Index(
            'ix_memory_cleanup',
            'session_id',
            'last_accessed',
            'importance',
            postgresql_where=text('deleted_at IS NULL')
        ),
        
        # Index for time-based queries
        Index(
            'ix_memory_created_at',
            'created_at'
        ),
        
        # ADDED: Check constraints for data integrity
        CheckConstraint(
            'importance >= 0.0 AND importance <= 1.0',
            name='ck_memory_importance_range'
        ),
        
        CheckConstraint(
            'access_count >= 0',
            name='ck_memory_access_count_positive'
        ),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<Memory(id={self.id}, session_id={self.session_id}, "
            f"type={self.content_type}, importance={self.importance}, "
            f"expired={self.is_expired()}, deleted={self.is_deleted()})>"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "session_id": self.session_id,
            "content_type": self.content_type,
            "content": self.content,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
            "importance": self.importance,
            "effective_importance": self.effective_importance,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_expired": self.is_expired(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "deleted_at": self.deleted_at.isoformat() if self.deleted_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """
        Create Memory instance from dictionary.
        
        Args:
            data: Dictionary with memory data
            
        Returns:
            Memory instance
        """
        # Convert ISO strings to datetime if needed
        datetime_fields = ['expires_at', 'last_accessed', 'created_at', 'updated_at', 'deleted_at']
        for field in datetime_fields:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)
    
    def update_access(self) -> None:
        """Update access tracking."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
    
    def validate(self) -> bool:
        """
        Validate memory data.
        
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        # Validate content_type
        valid_types = ['user_info', 'preference', 'fact', 'context']
        if self.content_type not in valid_types:
            raise ValueError(f"Invalid content_type. Must be one of: {valid_types}")
        
        # Validate importance range
        if not (0.0 <= self.importance <= 1.0):
            raise ValueError("Importance must be between 0.0 and 1.0")
        
        # Validate content not empty
        if not self.content or not self.content.strip():
            raise ValueError("Content cannot be empty")
        
        # Validate content_hash matches content
        expected_hash = self.compute_content_hash(self.content)
        if self.content_hash != expected_hash:
            raise ValueError("content_hash does not match content")
        
        # Validate metadata is JSON-serializable
        try:
            json.dumps(self.metadata)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Metadata must be JSON-serializable: {e}")
        
        return True
    
    # ===========================
    # Content Hashing (NEW)
    # ===========================
    
    @staticmethod
    def normalize_content(content: str) -> str:
        """
        Normalize content for consistent hashing.
        
        Normalization steps:
        1. Convert to lowercase
        2. Normalize whitespace to single spaces
        3. Strip trailing punctuation
        
        Args:
            content: Raw content string
            
        Returns:
            Normalized content
        """
        # Convert to lowercase
        normalized = content.lower()
        
        # Normalize whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove trailing punctuation
        normalized = normalized.rstrip('.,!?;:')
        
        return normalized
    
    @staticmethod
    def compute_content_hash(content: str) -> str:
        """
        Compute SHA256 hash of normalized content.
        
        Args:
            content: Content string
            
        Returns:
            Hex digest of content hash (64 characters)
        """
        normalized = Memory.normalize_content(content)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    @staticmethod
    def compute_semantic_hash(content: str, num_bits: int = 64) -> str:
        """
        Compute SimHash for semantic similarity detection.
        
        This is a simplified implementation. For production use,
        consider using a proper LSH library or sentence embeddings.
        
        Args:
            content: Content string
            num_bits: Hash size in bits (default 64)
            
        Returns:
            Hex digest of semantic hash
        """
        # Tokenize normalized content
        tokens = Memory.normalize_content(content).split()
        
        # Initialize bit vector
        v = [0] * num_bits
        
        # Process each token
        for token in tokens:
            # Hash token to get fingerprint
            h = hashlib.sha256(token.encode('utf-8')).digest()
            
            # Update bit vector based on fingerprint
            for i in range(num_bits):
                byte_idx = i // 8
                bit_idx = i % 8
                
                if byte_idx < len(h):
                    if (h[byte_idx] >> bit_idx) & 1:
                        v[i] += 1
                    else:
                        v[i] -= 1
        
        # Convert bit vector to binary fingerprint
        fingerprint = 0
        for i in range(num_bits):
            if v[i] > 0:
                fingerprint |= (1 << i)
        
        # Convert to hex
        hex_length = num_bits // 4
        return format(fingerprint, f'0{hex_length}x')
    
    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """
        Calculate Hamming distance between two hashes.
        
        Useful for finding similar memories based on semantic_hash.
        
        Args:
            hash1: First hash (hex string)
            hash2: Second hash (hex string)
            
        Returns:
            Hamming distance (number of differing bits)
        """
        # Convert hex to binary
        int1 = int(hash1, 16)
        int2 = int(hash2, 16)
        
        # XOR to find differing bits
        xor_result = int1 ^ int2
        
        # Count number of 1s (differing bits)
        distance = bin(xor_result).count('1')
        
        return distance
    
    # ===========================
    # Factory Methods (NEW)
    # ===========================
    
    @classmethod
    def create_memory(
        cls,
        session_id: str,
        content: str,
        content_type: str,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_hours: Optional[int] = None
    ) -> 'Memory':
        """
        Factory method to create memory with automatic hash computation.
        
        Args:
            session_id: Session identifier
            content: Memory content
            content_type: Memory type (user_info, preference, fact, context)
            importance: Importance score (0.0-1.0)
            metadata: Additional metadata
            ttl_hours: Time-to-live in hours (optional)
            
        Returns:
            Memory instance with computed hashes
        """
        # Compute hashes
        content_hash = cls.compute_content_hash(content)
        semantic_hash = cls.compute_semantic_hash(content)
        
        # Calculate expiration
        expires_at = None
        if ttl_hours:
            expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)
        
        return cls(
            id=str(uuid.uuid4()),
            session_id=session_id,
            content=content,
            content_type=content_type,
            content_hash=content_hash,
            semantic_hash=semantic_hash,
            importance=importance,
            metadata=metadata or {},
            expires_at=expires_at
        )
    
    # ===========================
    # Lifecycle Methods (NEW)
    # ===========================
    
    def is_expired(self) -> bool:
        """
        Check if memory has expired.
        
        Returns:
            True if expired, False otherwise
        """
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def is_deleted(self) -> bool:
        """
        Check if memory is soft-deleted.
        
        Returns:
            True if deleted, False otherwise
        """
        return self.deleted_at is not None
    
    def soft_delete(self) -> None:
        """
        Soft delete memory for GDPR compliance.
        
        Marks the memory as deleted without removing from database.
        Useful for audit trails and compliance.
        """
        self.deleted_at = datetime.utcnow()
    
    def restore(self) -> None:
        """
        Restore a soft-deleted memory.
        
        Removes the deleted_at timestamp, making the memory active again.
        """
        self.deleted_at = None
    
    def extend_ttl(self, hours: int) -> None:
        """
        Extend the TTL of the memory.
        
        Args:
            hours: Hours to extend from now
        """
        self.expires_at = datetime.utcnow() + timedelta(hours=hours)
    
    # ===========================
    # Importance Decay (NEW)
    # ===========================
    
    def calculate_decay_factor(self, half_life_days: int = 30) -> float:
        """
        Calculate importance decay factor based on age.
        
        Implements exponential decay: importance * e^(-λt)
        where λ = ln(2) / half_life
        
        This allows older memories to naturally become less important
        over time, unless they are frequently accessed.
        
        Args:
            half_life_days: Days for importance to decay by half
            
        Returns:
            Decay factor (0.0 to 1.0)
        """
        age_days = (datetime.utcnow() - self.created_at).days
        
        # Decay constant: ln(2) / half_life
        decay_constant = 0.693147 / half_life_days
        
        # Exponential decay: e^(-λt)
        decay_factor = math.exp(-decay_constant * age_days)
        
        return decay_factor
    
    @hybrid_property
    def effective_importance(self) -> float:
        """
        Calculate effective importance with decay.
        
        Combines base importance with time-based decay.
        Frequently accessed memories maintain higher importance.
        
        Returns:
            Decayed importance score
        """
        # Base importance
        base_importance = self.importance
        
        # Apply time-based decay
        decay = self.calculate_decay_factor(half_life_days=30)
        
        # Boost based on access frequency (up to 20% boost)
        access_boost = min(0.2, self.access_count * 0.02)
        
        # Combine: base * decay + access_boost
        effective = (base_importance * decay) + access_boost
        
        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, effective))


__all__ = ['Memory']
