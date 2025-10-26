# Phase 3: Tools Implementation - Meticulous Plan & Complete Code

## 📋 Implementation Plan

### Overview
We'll implement four tools that provide the core capabilities of our Customer Support AI Agent:
1. **RAG Tool** - Knowledge base search with EmbeddingGemma and ChromaDB
2. **Memory Tool** - Conversation memory with SQLite persistence
3. **Attachment Tool** - Document processing with MarkItDown
4. **Escalation Tool** - Human handoff detection and routing

### Dependencies & Structure
```
backend/app/tools/
├── __init__.py          # Tool exports
├── base_tool.py         # Base class for all tools
├── rag_tool.py          # RAG implementation
├── memory_tool.py       # Memory management
├── attachment_tool.py   # Document processing
└── escalation_tool.py   # Escalation logic

backend/app/services/
├── cache_service.py     # Redis caching
└── embedding_service.py # Embedding generation

backend/app/models/
├── memory.py           # Memory SQLAlchemy model
└── schemas.py          # Pydantic schemas
```

---

## 🛠️ Implementation Files

### File 1: Base Tool Class

**`backend/app/tools/__init__.py`**
```python
"""
Tools module for Customer Support AI Agent.
Exports all available tools for agent use.
"""

from .base_tool import BaseTool
from .rag_tool import RAGTool
from .memory_tool import MemoryTool
from .attachment_tool import AttachmentTool
from .escalation_tool import EscalationTool

__all__ = [
    "BaseTool",
    "RAGTool", 
    "MemoryTool",
    "AttachmentTool",
    "EscalationTool"
]
```

**`backend/app/tools/base_tool.py`**
```python
"""
Base tool class for Microsoft Agent Framework tools.
All tools should inherit from this base class.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Abstract base class for agent tools.
    Provides common interface and functionality for all tools.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize base tool.
        
        Args:
            name: Unique tool identifier
            description: Human-readable tool description
        """
        self.name = name
        self.description = description
        self.initialized = False
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize tool resources."""
        try:
            self._setup()
            self.initialized = True
            logger.info(f"Tool '{self.name}' initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize tool '{self.name}': {e}")
            raise
    
    @abstractmethod
    def _setup(self) -> None:
        """Setup tool-specific resources. Override in subclasses."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute tool action.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            Dictionary with execution results
        """
        pass
    
    async def __call__(self, **kwargs) -> Dict[str, Any]:
        """
        Make tool callable for agent framework.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            Dictionary with execution results
        """
        if not self.initialized:
            raise RuntimeError(f"Tool '{self.name}' not initialized")
        
        try:
            logger.debug(f"Executing tool '{self.name}' with params: {list(kwargs.keys())}")
            result = await self.execute(**kwargs)
            logger.debug(f"Tool '{self.name}' execution completed")
            return result
        except Exception as e:
            logger.error(f"Tool '{self.name}' execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": self.name
            }
    
    async def cleanup(self) -> None:
        """Cleanup tool resources. Override if needed."""
        pass
```

### File 2: Cache Service

**`backend/app/services/cache_service.py`**
```python
"""
Redis cache service for application-wide caching.
Provides async caching operations with TTL support.
"""
import json
import logging
from typing import Any, Optional, Union
import hashlib
from datetime import timedelta

try:
    import redis.asyncio as redis
except ImportError:
    import aioredis as redis  # Fallback for older versions

from ..config import settings

logger = logging.getLogger(__name__)


class CacheService:
    """
    Async Redis cache service with JSON serialization.
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
        
        if self.enabled:
            self._connect()
    
    def _connect(self) -> None:
        """Establish Redis connection."""
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
            return False
        
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
            return 0
        
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
            return 0
    
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
            return False
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            logger.info("Redis cache service closed")
```

### File 3: RAG Tool Implementation

**`backend/app/tools/rag_tool.py`**
```python
"""
RAG (Retrieval-Augmented Generation) tool implementation.
Uses EmbeddingGemma for embeddings and ChromaDB for vector storage.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from ..config import settings
from ..services.cache_service import CacheService
from .base_tool import BaseTool

logger = logging.getLogger(__name__)

# EmbeddingGemma-specific prefixes for optimal performance
QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: none | text: "

# Chunking parameters
CHUNK_SIZE = 500  # words
CHUNK_OVERLAP = 50  # words

# Search parameters
DEFAULT_K = 5
SIMILARITY_THRESHOLD = 0.7


class RAGTool(BaseTool):
    """
    RAG tool for searching and retrieving relevant documents.
    Uses Google's EmbeddingGemma model for generating embeddings
    and ChromaDB for efficient vector similarity search.
    """
    
    def __init__(self):
        """Initialize RAG tool with embedding model and vector store."""
        super().__init__(
            name="rag_search",
            description="Search knowledge base for relevant information using semantic similarity"
        )
        
    def _setup(self) -> None:
        """Setup embedding model, ChromaDB, and cache service."""
        # Initialize cache service
        self.cache = CacheService()
        
        # Initialize embedding model
        self._init_embedding_model()
        
        # Initialize ChromaDB
        self._init_chromadb()
    
    def _init_embedding_model(self) -> None:
        """Initialize EmbeddingGemma model with proper configuration."""
        try:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            
            # Load the SentenceTransformer model
            self.embedder = SentenceTransformer(
                settings.embedding_model,
                device='cpu'  # Use 'cuda' if GPU available
            )
            
            # Set the embedding dimension
            self.embedding_dim = settings.embedding_dimension
            
            logger.info(f"Embedding model loaded successfully (dim: {self.embedding_dim})")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback to a smaller model if EmbeddingGemma fails
            logger.warning("Falling back to all-MiniLM-L6-v2")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384
    
    def _init_chromadb(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            # Ensure persistence directory exists
            persist_dir = Path(settings.chroma_persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name=settings.chroma_collection_name
                )
                logger.info(f"Using existing ChromaDB collection: {settings.chroma_collection_name}")
                
            except ValueError:
                # Collection doesn't exist, create it
                self.collection = self.chroma_client.create_collection(
                    name=settings.chroma_collection_name,
                    metadata={
                        "hnsw:space": "ip",  # Inner product for normalized vectors
                        "hnsw:construction_ef": 200,
                        "hnsw:M": 16
                    }
                )
                logger.info(f"Created new ChromaDB collection: {settings.chroma_collection_name}")
                
                # Add sample documents if collection is empty
                self._add_sample_documents()
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _add_sample_documents(self) -> None:
        """Add sample documents to empty collection for testing."""
        sample_docs = [
            "To reset your password, click on 'Forgot Password' on the login page and follow the instructions.",
            "Our refund policy allows returns within 30 days of purchase for a full refund.",
            "Customer support is available 24/7 via chat, email at support@example.com, or phone at 1-800-EXAMPLE.",
            "To track your order, use the tracking number provided in your confirmation email.",
            "Account verification requires a valid email address and phone number for security purposes."
        ]
        
        try:
            self.add_documents(
                documents=sample_docs,
                metadatas=[{"type": "sample", "category": "faq"} for _ in sample_docs]
            )
            logger.info(f"Added {len(sample_docs)} sample documents to collection")
        except Exception as e:
            logger.warning(f"Failed to add sample documents: {e}")
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query text
            
        Returns:
            Normalized embedding vector
        """
        # Add query prefix for EmbeddingGemma
        prefixed_query = QUERY_PREFIX + query
        
        # Generate embedding
        embedding = self.embedder.encode(
            prefixed_query,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return embedding
    
    def embed_documents(self, documents: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of normalized embedding vectors
        """
        # Add document prefix for each document
        prefixed_docs = [DOC_PREFIX + doc for doc in documents]
        
        # Generate embeddings in batches
        embeddings = self.embedder.encode(
            prefixed_docs,
            normalize_embeddings=True,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=len(documents) > 10,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def chunk_document(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Split document into overlapping chunks.
        
        Args:
            text: Document text to chunk
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_words = words[i:i + CHUNK_SIZE]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_words) >= CHUNK_OVERLAP:  # Skip very small chunks
                metadata = {
                    "chunk_index": len(chunks),
                    "start_word": i,
                    "end_word": min(i + CHUNK_SIZE, len(words)),
                    "total_words": len(words)
                }
                chunks.append((chunk_text, metadata))
        
        return chunks
    
    async def search(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, Any]] = None,
        threshold: float = SIMILARITY_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Search for relevant documents using vector similarity.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            threshold: Minimum similarity threshold
            
        Returns:
            Search results with documents and metadata
        """
        # Create cache key
        cache_key = f"rag_search:{query}:{k}:{str(filter)}"
        
        # Check cache first
        if self.cache.enabled:
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_result
        
        try:
            # Generate query embedding
            query_embedding = self.embed_query(query)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format and filter results
            formatted_results = {
                "query": query,
                "sources": [],
                "total_results": 0
            }
            
            if results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    # Convert distance to similarity score (1 - distance for normalized vectors)
                    similarity = 1 - results['distances'][0][i]
                    
                    # Only include results above threshold
                    if similarity >= threshold:
                        source = {
                            "content": results['documents'][0][i],
                            "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                            "relevance_score": round(similarity, 4),
                            "rank": i + 1
                        }
                        formatted_results['sources'].append(source)
                
                formatted_results['total_results'] = len(formatted_results['sources'])
            
            # Cache the results
            if self.cache.enabled and formatted_results['total_results'] > 0:
                await self.cache.set(cache_key, formatted_results, ttl=settings.redis_ttl)
            
            logger.info(
                f"RAG search completed: query='{query[:50]}...', "
                f"results={formatted_results['total_results']}/{k}"
            )
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"RAG search error: {e}", exc_info=True)
            return {
                "query": query,
                "sources": [],
                "error": str(e)
            }
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        chunk: bool = True
    ) -> Dict[str, Any]:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
            ids: Optional IDs for documents
            chunk: Whether to chunk documents before adding
            
        Returns:
            Status and statistics of the operation
        """
        try:
            all_chunks = []
            all_metadatas = []
            all_ids = []
            
            for idx, doc in enumerate(documents):
                if chunk and len(doc.split()) > CHUNK_SIZE:
                    # Chunk large documents
                    chunks = self.chunk_document(doc)
                    for chunk_idx, (chunk_text, chunk_meta) in enumerate(chunks):
                        all_chunks.append(chunk_text)
                        
                        # Combine document metadata with chunk metadata
                        combined_meta = chunk_meta.copy()
                        if metadatas and idx < len(metadatas):
                            combined_meta.update(metadatas[idx])
                        combined_meta['doc_index'] = idx
                        all_metadatas.append(combined_meta)
                        
                        # Generate chunk ID
                        if ids and idx < len(ids):
                            chunk_id = f"{ids[idx]}_chunk_{chunk_idx}"
                        else:
                            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()
                        all_ids.append(chunk_id)
                else:
                    # Add document as-is
                    all_chunks.append(doc)
                    
                    meta = {"doc_index": idx}
                    if metadatas and idx < len(metadatas):
                        meta.update(metadatas[idx])
                    all_metadatas.append(meta)
                    
                    if ids and idx < len(ids):
                        all_ids.append(ids[idx])
                    else:
                        all_ids.append(hashlib.md5(doc.encode()).hexdigest())
            
            if not all_chunks:
                return {
                    "success": False,
                    "error": "No documents to add"
                }
            
            # Generate embeddings
            embeddings = self.embed_documents(all_chunks)
            
            # Add to ChromaDB
            self.collection.add(
                documents=all_chunks,
                embeddings=[emb.tolist() for emb in embeddings],
                metadatas=all_metadatas,
                ids=all_ids
            )
            
            # Clear cache as new documents were added
            if self.cache.enabled:
                asyncio.create_task(self.cache.clear_pattern("rag_search:*"))
            
            logger.info(
                f"Added {len(documents)} documents "
                f"({len(all_chunks)} chunks) to knowledge base"
            )
            
            return {
                "success": True,
                "documents_added": len(documents),
                "chunks_created": len(all_chunks)
            }
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute RAG search.
        
        Accepts:
            query: Search query (required)
            k: Number of results (optional, default: 5)
            filter: Metadata filter (optional)
            threshold: Similarity threshold (optional, default: 0.7)
            
        Returns:
            Search results with sources
        """
        query = kwargs.get("query")
        if not query:
            return {
                "success": False,
                "error": "Query parameter is required"
            }
        
        k = kwargs.get("k", DEFAULT_K)
        filter_dict = kwargs.get("filter")
        threshold = kwargs.get("threshold", SIMILARITY_THRESHOLD)
        
        return await self.search(query, k, filter_dict, threshold)
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, 'cache') and self.cache:
            await self.cache.close()
```

### File 4: Memory Tool Implementation

**`backend/app/models/memory.py`**
```python
"""
SQLAlchemy models for memory storage.
"""
from sqlalchemy import Column, String, Text, DateTime, Float, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class Memory(Base):
    """
    Memory storage model for conversation context and facts.
    """
    __tablename__ = "memories"
    
    # Primary key
    id = Column(String(36), primary_key=True)
    
    # Session association
    session_id = Column(String(36), nullable=False, index=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    
    # Content
    content_type = Column(String(50), nullable=False)  # 'user_info', 'preference', 'context', 'fact'
    content = Column(Text, nullable=False)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    
    # Importance scoring
    importance = Column(Float, default=0.5, nullable=False)  # 0.0 to 1.0
    access_count = Column(Integer, default=0)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_session_type', 'session_id', 'content_type'),
        Index('idx_session_importance', 'session_id', 'importance'),
        Index('idx_last_accessed', 'last_accessed'),
    )
    
    def __repr__(self):
        return f"<Memory(id={self.id}, session={self.session_id}, type={self.content_type})>"
```

**`backend/app/tools/memory_tool.py`**
```python
"""
Memory management tool for conversation context persistence.
Uses SQLite for storing and retrieving conversation memories.
"""
import logging
import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from contextlib import contextmanager

from sqlalchemy import create_engine, desc, and_, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from ..config import settings
from ..models.memory import Base, Memory
from .base_tool import BaseTool

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
    Provides persistent storage of important information across sessions.
    """
    
    def __init__(self):
        """Initialize memory tool with SQLite database."""
        super().__init__(
            name="memory_management",
            description="Store and retrieve conversation memory and context"
        )
    
    def _setup(self) -> None:
        """Setup SQLite database and create tables."""
        # Initialize database engine
        # Use StaticPool for SQLite to avoid threading issues
        self.engine = create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
            poolclass=StaticPool if "sqlite" in settings.database_url else None,
            echo=settings.database_echo
        )
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info(f"Memory tool initialized with database: {settings.database_url}")
    
    @contextmanager
    def get_db(self) -> Session:
        """
        Get database session with proper cleanup.
        
        Yields:
            Database session
        """
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    async def store_memory(
        self,
        session_id: str,
        content: str,
        content_type: str = "context",
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5
    ) -> Dict[str, Any]:
        """
        Store a memory entry for a session.
        
        Args:
            session_id: Session identifier
            content: Memory content to store
            content_type: Type of memory ('user_info', 'preference', 'context', 'fact')
            metadata: Optional metadata dictionary
            importance: Importance score (0.0 to 1.0)
            
        Returns:
            Status dictionary with memory ID
        """
        if content_type not in MEMORY_TYPE_PRIORITY:
            return {
                "success": False,
                "error": f"Invalid content_type. Must be one of: {list(MEMORY_TYPE_PRIORITY.keys())}"
            }
        
        if not (0.0 <= importance <= 1.0):
            importance = max(0.0, min(1.0, importance))
        
        try:
            with self.get_db() as db:
                # Check for duplicate memories (same content and type)
                existing = db.query(Memory).filter(
                    and_(
                        Memory.session_id == session_id,
                        Memory.content_type == content_type,
                        Memory.content == content
                    )
                ).first()
                
                if existing:
                    # Update importance and access time instead of creating duplicate
                    existing.importance = max(existing.importance, importance)
                    existing.last_accessed = datetime.utcnow()
                    existing.access_count += 1
                    db.commit()
                    
                    logger.debug(f"Updated existing memory: {existing.id}")
                    
                    return {
                        "success": True,
                        "memory_id": existing.id,
                        "action": "updated",
                        "message": "Memory updated successfully"
                    }
                
                # Create new memory
                memory = Memory(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    content_type=content_type,
                    content=content,
                    metadata=metadata or {},
                    importance=importance
                )
                
                db.add(memory)
                db.commit()
                
                logger.info(
                    f"Stored memory for session {session_id}: "
                    f"type={content_type}, importance={importance}"
                )
                
                return {
                    "success": True,
                    "memory_id": memory.id,
                    "action": "created",
                    "message": "Memory stored successfully"
                }
                
        except Exception as e:
            logger.error(f"Failed to store memory: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def retrieve_memories(
        self,
        session_id: str,
        content_type: Optional[str] = None,
        limit: int = DEFAULT_MEMORY_LIMIT,
        time_window_hours: Optional[int] = None,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories for a session.
        
        Args:
            session_id: Session identifier
            content_type: Filter by memory type (optional)
            limit: Maximum number of memories to retrieve
            time_window_hours: Only retrieve memories from last N hours (optional)
            min_importance: Minimum importance threshold
            
        Returns:
            List of memory dictionaries
        """
        try:
            with self.get_db() as db:
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
                    memory.last_accessed = datetime.utcnow()
                    memory.access_count += 1
                
                db.commit()
                
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
                
                logger.debug(
                    f"Retrieved {len(results)} memories for session {session_id}"
                )
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}", exc_info=True)
            return []
    
    async def summarize_session(
        self,
        session_id: str,
        max_items_per_type: int = 3
    ) -> str:
        """
        Generate a text summary of session memories.
        
        Args:
            session_id: Session identifier
            max_items_per_type: Maximum items per memory type
            
        Returns:
            Text summary of session context
        """
        try:
            # Retrieve memories grouped by type
            memory_groups = {}
            
            for content_type in MEMORY_TYPE_PRIORITY.keys():
                memories = await self.retrieve_memories(
                    session_id=session_id,
                    content_type=content_type,
                    limit=max_items_per_type,
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
    
    async def update_importance(
        self,
        memory_id: str,
        importance_delta: float
    ) -> Dict[str, Any]:
        """
        Update the importance score of a memory.
        
        Args:
            memory_id: Memory identifier
            importance_delta: Change in importance (-1.0 to 1.0)
            
        Returns:
            Status dictionary
        """
        try:
            with self.get_db() as db:
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                
                if not memory:
                    return {
                        "success": False,
                        "error": "Memory not found"
                    }
                
                # Update importance (keep within bounds)
                new_importance = max(0.0, min(1.0, memory.importance + importance_delta))
                memory.importance = new_importance
                memory.updated_at = datetime.utcnow()
                
                db.commit()
                
                logger.debug(f"Updated memory importance: {memory_id} -> {new_importance}")
                
                return {
                    "success": True,
                    "memory_id": memory_id,
                    "new_importance": new_importance
                }
                
        except Exception as e:
            logger.error(f"Failed to update memory importance: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def cleanup_old_memories(
        self,
        days: int = 30,
        max_per_session: int = 100
    ) -> Dict[str, Any]:
        """
        Clean up old and low-importance memories.
        
        Args:
            days: Delete memories older than N days with low importance
            max_per_session: Maximum memories to keep per session
            
        Returns:
            Cleanup statistics
        """
        try:
            with self.get_db() as db:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Delete old, low-importance, rarely accessed memories
                deleted_old = db.query(Memory).filter(
                    and_(
                        Memory.last_accessed < cutoff_date,
                        Memory.importance < 0.3,
                        Memory.access_count < 3
                    )
                ).delete()
                
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
                
                db.commit()
                
                total_deleted = deleted_old + deleted_excess
                
                logger.info(
                    f"Memory cleanup completed: {total_deleted} memories deleted "
                    f"({deleted_old} old, {deleted_excess} excess)"
                )
                
                return {
                    "success": True,
                    "deleted_old": deleted_old,
                    "deleted_excess": deleted_excess,
                    "total_deleted": total_deleted
                }
                
        except Exception as e:
            logger.error(f"Failed to cleanup memories: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def execute(self, **kwargs) -> Any:
        """
        Execute memory operations.
        
        Accepts:
            action: Operation to perform ('store', 'retrieve', 'summarize')
            session_id: Session identifier (required)
            Other parameters based on action
            
        Returns:
            Operation results
        """
        action = kwargs.get("action", "retrieve")
        session_id = kwargs.get("session_id")
        
        if not session_id:
            return {
                "success": False,
                "error": "session_id is required"
            }
        
        if action == "store":
            content = kwargs.get("content")
            if not content:
                return {
                    "success": False,
                    "error": "content is required for store action"
                }
            
            return await self.store_memory(
                session_id=session_id,
                content=content,
                content_type=kwargs.get("content_type", "context"),
                metadata=kwargs.get("metadata"),
                importance=kwargs.get("importance", 0.5)
            )
        
        elif action == "retrieve":
            memories = await self.retrieve_memories(
                session_id=session_id,
                content_type=kwargs.get("content_type"),
                limit=kwargs.get("limit", DEFAULT_MEMORY_LIMIT),
                time_window_hours=kwargs.get("time_window_hours"),
                min_importance=kwargs.get("min_importance", 0.0)
            )
            
            return {
                "success": True,
                "memories": memories,
                "count": len(memories)
            }
        
        elif action == "summarize":
            summary = await self.summarize_session(
                session_id=session_id,
                max_items_per_type=kwargs.get("max_items_per_type", 3)
            )
            
            return {
                "success": True,
                "summary": summary
            }
        
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}. Valid actions: store, retrieve, summarize"
            }
```

### File 5: Attachment Tool Implementation

**`backend/app/tools/attachment_tool.py`**
```python
"""
Attachment processing tool using MarkItDown.
Handles various file formats and extracts content for analysis.
"""
import logging
import os
import tempfile
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
import aiofiles
import asyncio

try:
    from markitdown import MarkItDown
except ImportError:
    logger.warning("MarkItDown not installed. Attachment processing will be limited.")
    MarkItDown = None

from ..config import settings
from .base_tool import BaseTool

logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    # Documents
    '.pdf': 'PDF document',
    '.docx': 'Word document',
    '.doc': 'Word document',
    '.xlsx': 'Excel spreadsheet',
    '.xls': 'Excel spreadsheet',
    '.pptx': 'PowerPoint presentation',
    '.ppt': 'PowerPoint presentation',
    
    # Text
    '.txt': 'Text file',
    '.md': 'Markdown file',
    '.rtf': 'Rich text file',
    '.csv': 'CSV file',
    '.json': 'JSON file',
    '.xml': 'XML file',
    '.yaml': 'YAML file',
    '.yml': 'YAML file',
    
    # Web
    '.html': 'HTML file',
    '.htm': 'HTML file',
    
    # Images (OCR if available)
    '.jpg': 'JPEG image',
    '.jpeg': 'JPEG image',
    '.png': 'PNG image',
    '.gif': 'GIF image',
    '.bmp': 'Bitmap image',
    
    # Audio (transcription if available)
    '.mp3': 'MP3 audio',
    '.wav': 'WAV audio',
    '.m4a': 'M4A audio',
    '.ogg': 'OGG audio',
}


class AttachmentTool(BaseTool):
    """
    Tool for processing file attachments and extracting content.
    Uses MarkItDown to convert various formats to readable text.
    """
    
    def __init__(self):
        """Initialize attachment processing tool."""
        super().__init__(
            name="attachment_processor",
            description="Process and extract content from uploaded files"
        )
    
    def _setup(self) -> None:
        """Setup MarkItDown and temporary directory."""
        # Initialize MarkItDown if available
        if MarkItDown:
            try:
                self.markitdown = MarkItDown(
                    enable_plugins=True  # Enable all available plugins
                )
                logger.info("MarkItDown initialized with all plugins")
            except Exception as e:
                logger.warning(f"Failed to initialize MarkItDown with plugins: {e}")
                self.markitdown = MarkItDown() if MarkItDown else None
        else:
            self.markitdown = None
            logger.warning("MarkItDown not available. Limited file processing enabled.")
        
        # Create temporary directory for file processing
        self.temp_dir = Path(tempfile.gettempdir()) / "cs_agent_attachments"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Attachment temp directory: {self.temp_dir}")
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get basic information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File information dictionary
        """
        path = Path(file_path)
        
        if not path.exists():
            return {
                "exists": False,
                "error": "File not found"
            }
        
        stat = path.stat()
        extension = path.suffix.lower()
        
        return {
            "exists": True,
            "filename": path.name,
            "extension": extension,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "file_type": SUPPORTED_EXTENSIONS.get(extension, "Unknown"),
            "supported": extension in SUPPORTED_EXTENSIONS,
            "modified": stat.st_mtime
        }
    
    async def save_uploaded_file(
        self,
        file_data: bytes,
        filename: str
    ) -> str:
        """
        Save uploaded file to temporary location.
        
        Args:
            file_data: File content as bytes
            filename: Original filename
            
        Returns:
            Path to saved file
        """
        # Generate unique filename
        file_hash = hashlib.md5(file_data).hexdigest()[:8]
        safe_filename = f"{file_hash}_{Path(filename).name}"
        temp_path = self.temp_dir / safe_filename
        
        try:
            # Write file asynchronously
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(file_data)
            
            logger.info(f"Saved uploaded file: {temp_path}")
            return str(temp_path)
            
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {e}")
            raise
    
    def process_with_markitdown(
        self,
        file_path: str
    ) -> Optional[str]:
        """
        Process file with MarkItDown.
        
        Args:
            file_path: Path to file
            
        Returns:
            Extracted text content or None if failed
        """
        if not self.markitdown:
            return None
        
        try:
            result = self.markitdown.convert(file_path)
            
            # Extract text content based on result type
            if hasattr(result, 'text_content'):
                content = result.text_content
            elif hasattr(result, 'content'):
                content = result.content
            elif isinstance(result, str):
                content = result
            else:
                content = str(result)
            
            return content
            
        except Exception as e:
            logger.error(f"MarkItDown processing failed: {e}")
            return None
    
    def process_text_file(self, file_path: str) -> Optional[str]:
        """
        Process plain text files.
        
        Args:
            file_path: Path to text file
            
        Returns:
            File content or None if failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read text file: {e}")
            return None
    
    def chunk_content(
        self,
        content: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[str]:
        """
        Split content into overlapping chunks for processing.
        
        Args:
            content: Text content to chunk
            chunk_size: Size of each chunk in words
            overlap: Overlap between chunks in words
            
        Returns:
            List of text chunks
        """
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    async def process_attachment(
        self,
        file_path: str,
        filename: Optional[str] = None,
        extract_metadata: bool = True,
        chunk_for_rag: bool = False
    ) -> Dict[str, Any]:
        """
        Process an attachment and extract content.
        
        Args:
            file_path: Path to the file
            filename: Original filename (optional)
            extract_metadata: Whether to extract file metadata
            chunk_for_rag: Whether to chunk content for RAG indexing
            
        Returns:
            Processing results with extracted content
        """
        # Get file info
        file_info = self.get_file_info(file_path)
        
        if not file_info["exists"]:
            return {
                "success": False,
                "error": "File not found",
                "file_path": file_path
            }
        
        # Use provided filename or extract from path
        if not filename:
            filename = file_info["filename"]
        
        # Check if file is supported
        if not file_info["supported"]:
            return {
                "success": False,
                "error": f"Unsupported file type: {file_info['extension']}",
                "supported_types": list(SUPPORTED_EXTENSIONS.keys()),
                "filename": filename
            }
        
        # Check file size
        if file_info["size_bytes"] > settings.max_file_size:
            return {
                "success": False,
                "error": f"File too large. Max size: {settings.max_file_size / (1024*1024)}MB",
                "filename": filename,
                "size_mb": file_info["size_mb"]
            }
        
        try:
            # Try MarkItDown first
            content = self.process_with_markitdown(file_path)
            
            # Fallback for text files
            if not content and file_info["extension"] in ['.txt', '.md', '.csv', '.json', '.xml']:
                content = self.process_text_file(file_path)
            
            if not content:
                return {
                    "success": False,
                    "error": "Failed to extract content from file",
                    "filename": filename
                }
            
            # Build response
            result = {
                "success": True,
                "filename": filename,
                "file_type": file_info["file_type"],
                "extension": file_info["extension"],
                "size_mb": file_info["size_mb"],
                "content": content,
                "content_length": len(content),
                "word_count": len(content.split())
            }
            
            # Add metadata if requested
            if extract_metadata:
                result["metadata"] = {
                    "processed_with": "MarkItDown" if self.markitdown else "fallback",
                    "file_path": file_path,
                    "original_size": file_info["size_bytes"]
                }
            
            # Add chunks if requested
            if chunk_for_rag and len(content.split()) > 500:
                chunks = self.chunk_content(content)
                result["chunks"] = chunks
                result["chunk_count"] = len(chunks)
            
            # Add preview
            preview_length = min(500, len(content))
            result["preview"] = content[:preview_length] + ("..." if len(content) > preview_length else "")
            
            # Add specific notes based on file type
            if file_info["extension"] in ['.xlsx', '.xls', '.csv']:
                result["note"] = "Tabular data extracted and converted to text format"
            elif file_info["extension"] in ['.jpg', '.jpeg', '.png']:
                result["note"] = "Image processed for text extraction (OCR)"
            elif file_info["extension"] in ['.pdf']:
                result["note"] = "PDF content extracted, formatting may vary"
            elif file_info["extension"] in ['.docx', '.doc']:
                result["note"] = "Word document converted to plain text"
            
            logger.info(
                f"Successfully processed attachment: {filename} "
                f"({result['word_count']} words extracted)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing attachment {filename}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "filename": filename
            }
    
    async def process_multiple(
        self,
        file_paths: List[str],
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Process multiple attachments concurrently.
        
        Args:
            file_paths: List of file paths
            max_concurrent: Maximum concurrent processing
            
        Returns:
            List of processing results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(file_path):
            async with semaphore:
                return await self.process_attachment(file_path)
        
        tasks = [process_with_semaphore(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def extract_and_index(
        self,
        file_path: str,
        filename: str,
        rag_tool: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Extract content and optionally index in RAG system.
        
        Args:
            file_path: Path to file
            filename: Original filename
            rag_tool: RAG tool instance for indexing
            
        Returns:
            Processing and indexing results
        """
        # Process the attachment
        result = await self.process_attachment(
            file_path,
            filename,
            chunk_for_rag=True
        )
        
        if not result["success"]:
            return result
        
        # Index in RAG if tool provided and chunks available
        if rag_tool and "chunks" in result:
            try:
                # Prepare metadata for each chunk
                metadatas = [
                    {
                        "source": filename,
                        "chunk_index": i,
                        "file_type": result["file_type"],
                        "total_chunks": result["chunk_count"]
                    }
                    for i in range(result["chunk_count"])
                ]
                
                # Add to RAG
                index_result = rag_tool.add_documents(
                    documents=result["chunks"],
                    metadatas=metadatas
                )
                
                result["indexed"] = index_result.get("success", False)
                result["documents_indexed"] = index_result.get("chunks_created", 0)
                
                if result["indexed"]:
                    logger.info(f"Indexed {result['documents_indexed']} chunks from {filename}")
                
            except Exception as e:
                logger.error(f"Failed to index attachment in RAG: {e}")
                result["indexed"] = False
                result["index_error"] = str(e)
        
        return result
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old temporary files.
        
        Args:
            max_age_hours: Delete files older than this
            
        Returns:
            Number of files deleted
        """
        import time
        
        deleted = 0
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        try:
            for file_path in self.temp_dir.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted += 1
            
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old attachment files")
            
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")
        
        return deleted
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute attachment processing.
        
        Accepts:
            file_path: Path to file (required)
            filename: Original filename (optional)
            extract_metadata: Whether to include metadata (default: True)
            chunk_for_rag: Whether to chunk for RAG (default: False)
            
        Returns:
            Processing results
        """
        file_path = kwargs.get("file_path")
        
        if not file_path:
            return {
                "success": False,
                "error": "file_path is required"
            }
        
        return await self.process_attachment(
            file_path=file_path,
            filename=kwargs.get("filename"),
            extract_metadata=kwargs.get("extract_metadata", True),
            chunk_for_rag=kwargs.get("chunk_for_rag", False)
        )
    
    async def cleanup(self) -> None:
        """Cleanup temporary files on shutdown."""
        deleted = self.cleanup_temp_files(max_age_hours=0)
        logger.info(f"Attachment tool cleanup: {deleted} files removed")
```

### File 6: Escalation Tool Implementation

**`backend/app/tools/escalation_tool.py`**
```python
"""
Escalation tool for detecting when human intervention is needed.
Analyzes conversation context to determine escalation requirements.
"""
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..config import settings
from .base_tool import BaseTool

logger = logging.getLogger(__name__)

# Escalation trigger keywords and phrases
ESCALATION_KEYWORDS = {
    "urgent": 1.0,
    "emergency": 1.0,
    "complaint": 0.9,
    "angry": 0.9,
    "frustrated": 0.8,
    "disappointed": 0.8,
    "unacceptable": 0.8,
    "legal": 0.9,
    "lawsuit": 1.0,
    "lawyer": 0.9,
    "sue": 0.9,
    "refund": 0.7,
    "compensation": 0.7,
    "manager": 0.8,
    "supervisor": 0.8,
    "human": 0.7,
    "speak to someone": 0.8,
    "talk to a person": 0.8,
    "not helping": 0.7,
    "doesn't work": 0.6,
    "broken": 0.6,
    "critical": 0.9,
    "immediate": 0.8,
    "asap": 0.8,
    "right now": 0.8
}

# Sentiment thresholds
NEGATIVE_SENTIMENT_THRESHOLD = -0.5
ESCALATION_CONFIDENCE_THRESHOLD = 0.7


class EscalationTool(BaseTool):
    """
    Tool for detecting when a conversation should be escalated to human support.
    Analyzes various signals including keywords, sentiment, and context.
    """
    
    def __init__(self):
        """Initialize escalation detection tool."""
        super().__init__(
            name="escalation_check
