# 🚀 COMPLETE FILE IMPLEMENTATIONS (Final Phase)

## File 8: `backend/app/config/tool_settings.py` (Validated - Complete Replacement)

```python
"""
Tool-specific configuration settings.
Defines feature flags and per-tool configurations for the agent system.

Version: 2.0.0 (Validated and enhanced)
"""
from typing import Dict, Any, Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

import logging

logger = logging.getLogger(__name__)


class ToolSettings(BaseSettings):
    """
    Tool-specific configuration with feature flags.
    Each tool can be enabled/disabled and configured independently.
    
    Version 2.0.0: Validated all settings and improved documentation.
    """
    
    # ===========================
    # Tool Feature Flags
    # ===========================
    
    enable_rag_tool: bool = Field(
        default=True,
        description="Enable RAG (Retrieval-Augmented Generation) tool"
    )
    
    enable_memory_tool: bool = Field(
        default=True,
        description="Enable Memory management tool"
    )
    
    enable_escalation_tool: bool = Field(
        default=True,
        description="Enable Escalation detection tool"
    )
    
    enable_attachment_tool: bool = Field(
        default=True,
        description="Enable Attachment processing tool"
    )
    
    # Future tools (disabled by default)
    enable_crm_tool: bool = Field(
        default=False,
        description="Enable CRM lookup tool"
    )
    
    enable_billing_tool: bool = Field(
        default=False,
        description="Enable Billing/invoice tool"
    )
    
    enable_inventory_tool: bool = Field(
        default=False,
        description="Enable Inventory lookup tool"
    )
    
    # ===========================
    # RAG Tool Configuration
    # ===========================
    
    rag_chunk_size: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="RAG document chunk size in words"
    )
    
    rag_chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between chunks in words"
    )
    
    rag_search_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Default number of RAG search results"
    )
    
    rag_similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for RAG results"
    )
    
    rag_cache_enabled: bool = Field(
        default=True,
        description="Enable caching for RAG search results"
    )
    
    rag_cache_ttl: int = Field(
        default=3600,
        ge=60,
        description="RAG cache TTL in seconds"
    )
    
    # ===========================
    # Memory Tool Configuration
    # ===========================
    
    memory_max_entries: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum memory entries per session"
    )
    
    memory_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=720,
        description="Memory TTL in hours"
    )
    
    memory_cleanup_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days before cleaning old memories"
    )
    
    memory_importance_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum importance for memory retrieval"
    )
    
    # ===========================
    # Escalation Tool Configuration
    # ===========================
    
    escalation_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for escalation"
    )
    
    escalation_keywords: Dict[str, float] = Field(
        default_factory=lambda: {
            "urgent": 1.0,
            "emergency": 1.0,
            "complaint": 0.9,
            "legal": 0.9,
            "lawsuit": 1.0,
            "manager": 0.8,
            "supervisor": 0.8
        },
        description="Escalation keywords with weights"
    )
    
    escalation_notification_enabled: bool = Field(
        default=False,
        description="Enable automatic escalation notifications"
    )
    
    escalation_notification_email: Optional[str] = Field(
        default=None,
        description="Email address for escalation notifications"
    )
    
    escalation_notification_webhook: Optional[str] = Field(
        default=None,
        description="Webhook URL for escalation notifications"
    )
    
    # ===========================
    # Attachment Tool Configuration
    # ===========================
    
    attachment_max_file_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        ge=1024,
        description="Maximum attachment file size in bytes"
    )
    
    attachment_allowed_extensions: List[str] = Field(
        default_factory=lambda: [
            ".pdf", ".docx", ".doc", ".txt", ".md",
            ".csv", ".xlsx", ".xls", ".json", ".xml",
            ".jpg", ".jpeg", ".png"
        ],
        description="Allowed file extensions for attachments"
    )
    
    attachment_chunk_for_rag: bool = Field(
        default=True,
        description="Automatically chunk attachments for RAG indexing"
    )
    
    attachment_temp_cleanup_hours: int = Field(
        default=24,
        ge=1,
        description="Hours before cleaning up temporary attachment files"
    )
    
    # ===========================
    # CRM Tool Configuration
    # ===========================
    
    crm_api_endpoint: Optional[str] = Field(
        default=None,
        description="CRM API endpoint URL"
    )
    
    crm_api_key: Optional[str] = Field(
        default=None,
        description="CRM API key (use secrets manager in production)"
    )
    
    crm_timeout: int = Field(
        default=10,
        ge=1,
        description="CRM API timeout in seconds"
    )
    
    crm_max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum CRM API retry attempts"
    )
    
    # ===========================
    # Validators
    # ===========================
    
    @field_validator('escalation_keywords', mode='before')
    @classmethod
    def parse_escalation_keywords(cls, v):
        """Parse escalation keywords from various formats."""
        if v is None:
            return {
                "urgent": 1.0,
                "emergency": 1.0,
                "complaint": 0.9,
                "legal": 0.9,
                "lawsuit": 1.0,
                "manager": 0.8,
                "supervisor": 0.8
            }
        
        if isinstance(v, dict):
            return v
        
        if isinstance(v, str):
            import json
            # Try to parse as JSON
            if v.startswith('{'):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            
            # Parse as comma-separated key=value pairs
            result = {}
            for pair in v.split(','):
                if '=' in pair:
                    key, value = pair.strip().split('=', 1)
                    try:
                        result[key] = float(value)
                    except ValueError:
                        result[key] = 0.8
                else:
                    result[pair.strip()] = 0.8
            return result if result else cls.parse_escalation_keywords(None)
        
        return v
    
    @field_validator('attachment_allowed_extensions', mode='before')
    @classmethod
    def parse_allowed_extensions(cls, v):
        """Parse allowed extensions from various formats."""
        default = [
            ".pdf", ".docx", ".doc", ".txt", ".md",
            ".csv", ".xlsx", ".xls", ".json", ".xml",
            ".jpg", ".jpeg", ".png"
        ]
        
        if v is None:
            return default
        
        if isinstance(v, list):
            return v
        
        if isinstance(v, str):
            import json
            # Try to parse as JSON
            if v.startswith('['):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            
            # Parse as comma-separated
            return [ext.strip() for ext in v.split(',') if ext.strip()]
        
        return default
    
    # ===========================
    # Helper Methods
    # ===========================
    
    def get_enabled_tools(self) -> List[str]:
        """
        Get list of enabled tool names.
        
        Returns:
            List of enabled tool identifiers
        """
        enabled = []
        
        if self.enable_rag_tool:
            enabled.append('rag')
        if self.enable_memory_tool:
            enabled.append('memory')
        if self.enable_escalation_tool:
            enabled.append('escalation')
        if self.enable_attachment_tool:
            enabled.append('attachment')
        if self.enable_crm_tool:
            enabled.append('crm')
        if self.enable_billing_tool:
            enabled.append('billing')
        if self.enable_inventory_tool:
            enabled.append('inventory')
        
        return enabled
    
    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific tool.
        
        Args:
            tool_name: Tool identifier ('rag', 'memory', etc.)
            
        Returns:
            Dictionary of tool-specific configuration
        """
        if tool_name == 'rag':
            return {
                'chunk_size': self.rag_chunk_size,
                'chunk_overlap': self.rag_chunk_overlap,
                'search_k': self.rag_search_k,
                'similarity_threshold': self.rag_similarity_threshold,
                'cache_enabled': self.rag_cache_enabled,
                'cache_ttl': self.rag_cache_ttl
            }
        
        elif tool_name == 'memory':
            return {
                'max_entries': self.memory_max_entries,
                'ttl_hours': self.memory_ttl_hours,
                'cleanup_days': self.memory_cleanup_days,
                'importance_threshold': self.memory_importance_threshold
            }
        
        elif tool_name == 'escalation':
            return {
                'confidence_threshold': self.escalation_confidence_threshold,
                'keywords': self.escalation_keywords,
                'notification_enabled': self.escalation_notification_enabled,
                'notification_email': self.escalation_notification_email,
                'notification_webhook': self.escalation_notification_webhook
            }
        
        elif tool_name == 'attachment':
            return {
                'max_file_size': self.attachment_max_file_size,
                'allowed_extensions': self.attachment_allowed_extensions,
                'chunk_for_rag': self.attachment_chunk_for_rag,
                'temp_cleanup_hours': self.attachment_temp_cleanup_hours
            }
        
        elif tool_name == 'crm':
            return {
                'api_endpoint': self.crm_api_endpoint,
                'api_key': self.crm_api_key,
                'timeout': self.crm_timeout,
                'max_retries': self.crm_max_retries
            }
        
        else:
            return {}
    
    def validate_tool_config(self, tool_name: str) -> List[str]:
        """
        Validate configuration for a specific tool.
        
        Args:
            tool_name: Tool identifier
            
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        if tool_name == 'crm' and self.enable_crm_tool:
            if not self.crm_api_endpoint:
                warnings.append("CRM tool enabled but no API endpoint configured")
            if not self.crm_api_key:
                warnings.append("CRM tool enabled but no API key configured")
        
        if tool_name == 'escalation' and self.escalation_notification_enabled:
            if not self.escalation_notification_email and not self.escalation_notification_webhook:
                warnings.append(
                    "Escalation notifications enabled but no email or webhook configured"
                )
        
        return warnings


# Create global instance
tool_settings = ToolSettings()

# Export
__all__ = ['ToolSettings', 'tool_settings']
```

---

## File 9: `backend/app/tools/rag_tool.py` (Error Handling Fixed - Complete Replacement)

**Note: This file is quite large. I'll include the critical fixes with proper error handling.**

```python
"""
RAG (Retrieval-Augmented Generation) tool implementation.
Uses SentenceTransformer for embeddings and ChromaDB for vector storage.

Version: 2.0.0 (Enhanced error handling and cache safety)
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import numpy as np
from pathlib import Path
import asyncio

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config import settings
from ..services.cache_service import CacheService
from .base_tool import BaseTool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)

# Embedding prefixes for optimal performance
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
    
    Version 2.0.0:
    - Enhanced error handling for ChromaDB operations
    - Fixed cache task error handling
    - Improved null safety checks
    - Adaptive batch sizing
    """
    
    def __init__(self):
        """Initialize RAG tool with embedding model and vector store."""
        super().__init__(
            name="rag_search",
            description="Search knowledge base for relevant information using semantic similarity",
            version="2.0.0"
        )
        
        # Resources initialized in async initialize()
        self.embedder = None
        self.chroma_client = None
        self.collection = None
        self.cache = None
    
    async def initialize(self) -> None:
        """Initialize RAG tool resources (async-safe)."""
        try:
            logger.info(f"Initializing RAG tool '{self.name}'...")
            
            # Initialize cache service
            self.cache = CacheService()
            
            # Initialize embedding model (CPU-bound, run in thread pool)
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._init_embedding_model
            )
            
            # Initialize ChromaDB (I/O-bound, run in thread pool)
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._init_chromadb
            )
            
            self.initialized = True
            logger.info(f"✓ RAG tool '{self.name}' initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG tool: {e}", exc_info=True)
            raise
    
    async def cleanup(self) -> None:
        """Cleanup RAG tool resources."""
        try:
            logger.info(f"Cleaning up RAG tool '{self.name}'...")
            
            # Close cache connections
            if self.cache:
                await self.cache.close()
            
            # ChromaDB cleanup
            if self.chroma_client:
                self.chroma_client = None
            
            self.initialized = False
            logger.info(f"✓ RAG tool '{self.name}' cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during RAG tool cleanup: {e}")
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute RAG search (async-first)."""
        query = kwargs.get("query")
        if not query:
            return ToolResult.error_result(
                error="Query parameter is required",
                metadata={"tool": self.name}
            )
        
        k = kwargs.get("k", DEFAULT_K)
        filter_dict = kwargs.get("filter")
        threshold = kwargs.get("threshold", SIMILARITY_THRESHOLD)
        
        try:
            result = await self.search_async(query, k, filter_dict, threshold)
            
            return ToolResult.success_result(
                data=result,
                metadata={
                    "tool": self.name,
                    "query_length": len(query),
                    "k": k,
                    "threshold": threshold,
                    "results_count": result.get('total_results', 0)
                }
            )
            
        except Exception as e:
            logger.error(f"RAG execute error: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={"tool": self.name, "query": query[:100]}
            )
    
    async def search_async(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, Any]] = None,
        threshold: float = SIMILARITY_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Search for relevant documents using vector similarity (async).
        
        Version 2.0.0: FIXED - Enhanced error handling for ChromaDB operations.
        """
        # Create cache key
        cache_key = f"rag_search:{query}:{k}:{str(filter)}"
        
        # Check cache first
        if self.cache and self.cache.enabled:
            try:
                cached_result = await self.cache.get(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for query: {query[:50]}...")
                    return cached_result
            except Exception as e:
                logger.warning(f"Cache retrieval failed: {e}")
                # Continue without cache
        
        try:
            # Generate query embedding (CPU-bound, run in thread pool)
            try:
                query_embedding = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.embed_query,
                    query
                )
            except Exception as e:
                logger.error(f"Failed to generate query embedding: {e}")
                return {
                    "query": query,
                    "sources": [],
                    "total_results": 0,
                    "error": f"Embedding generation failed: {str(e)}"
                }
            
            # FIXED: Search in ChromaDB with proper error handling
            try:
                results = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.collection.query(
                        query_embeddings=[query_embedding.tolist()],
                        n_results=k,
                        where=filter,
                        include=["documents", "metadatas", "distances"]
                    )
                )
            except Exception as e:
                logger.error(f"ChromaDB query failed: {e}", exc_info=True)
                return {
                    "query": query,
                    "sources": [],
                    "total_results": 0,
                    "error": f"Vector search failed: {str(e)}"
                }
            
            # Format and filter results
            formatted_results = {
                "query": query,
                "sources": [],
                "total_results": 0
            }
            
            if results and results.get('documents') and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    # Convert distance to similarity score
                    similarity = 1 - results['distances'][0][i]
                    
                    # Only include results above threshold
                    if similarity >= threshold:
                        source = {
                            "content": results['documents'][0][i],
                            "metadata": results['metadatas'][0][i] if results.get('metadatas') else {},
                            "relevance_score": round(similarity, 4),
                            "rank": i + 1
                        }
                        formatted_results['sources'].append(source)
                
                formatted_results['total_results'] = len(formatted_results['sources'])
            
            # FIXED: Cache results with proper error handling
            if self.cache and self.cache.enabled and formatted_results['total_results'] > 0:
                # Use asyncio.create_task with error handling
                async def safe_cache_set():
                    try:
                        await self.cache.set(cache_key, formatted_results, ttl=settings.redis_ttl)
                    except Exception as e:
                        logger.error(f"Failed to cache RAG results: {e}")
                
                asyncio.create_task(safe_cache_set())
            
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
                "total_results": 0,
                "error": str(e)
            }
    
    async def add_documents_async(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        chunk: bool = True
    ) -> ToolResult:
        """Add documents to the knowledge base (async)."""
        try:
            # Prepare documents
            prep_result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._prepare_documents,
                documents,
                metadatas,
                ids,
                chunk
            )
            
            if not prep_result['chunks']:
                return ToolResult.error_result(
                    error="No documents to add",
                    metadata={"tool": self.name}
                )
            
            all_chunks = prep_result['chunks']
            all_metadatas = prep_result['metadatas']
            all_ids = prep_result['ids']
            
            # Generate embeddings with adaptive batching
            try:
                embeddings = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.embed_documents,
                    all_chunks
                )
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")
                return ToolResult.error_result(
                    error=f"Embedding generation failed: {str(e)}",
                    metadata={"tool": self.name}
                )
            
            # Add to ChromaDB with error handling
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.collection.add(
                        documents=all_chunks,
                        embeddings=[emb.tolist() for emb in embeddings],
                        metadatas=all_metadatas,
                        ids=all_ids
                    )
                )
            except Exception as e:
                logger.error(f"Failed to add documents to ChromaDB: {e}")
                return ToolResult.error_result(
                    error=f"Failed to index documents: {str(e)}",
                    metadata={"tool": self.name}
                )
            
            # FIXED: Clear cache with proper error handling
            if self.cache and self.cache.enabled:
                async def safe_cache_clear():
                    try:
                        await self.cache.clear_pattern("rag_search:*")
                    except Exception as e:
                        logger.error(f"Failed to clear RAG cache: {e}")
                
                asyncio.create_task(safe_cache_clear())
            
            logger.info(
                f"Added {len(documents)} documents "
                f"({len(all_chunks)} chunks) to knowledge base"
            )
            
            return ToolResult.success_result(
                data={
                    "documents_added": len(documents),
                    "chunks_created": len(all_chunks)
                },
                metadata={
                    "tool": self.name,
                    "chunking_enabled": chunk
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={"tool": self.name, "document_count": len(documents)}
            )
    
    # ===========================
    # Private Helper Methods
    # ===========================
    
    def _init_embedding_model(self) -> None:
        """Initialize embedding model (sync)."""
        try:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            
            self.embedder = SentenceTransformer(
                settings.embedding_model,
                device='cpu'
            )
            
            self.embedding_dim = settings.embedding_dimension
            logger.info(f"Embedding model loaded successfully (dim: {self.embedding_dim})")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            logger.warning("Falling back to all-MiniLM-L6-v2")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384
    
    def _init_chromadb(self) -> None:
        """Initialize ChromaDB client and collection (sync)."""
        try:
            persist_dir = Path(settings.chroma_persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            try:
                self.collection = self.chroma_client.get_collection(
                    name=settings.chroma_collection_name
                )
                logger.info(f"Using existing ChromaDB collection: {settings.chroma_collection_name}")
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name=settings.chroma_collection_name,
                    metadata={
                        "hnsw:space": "ip",
                        "hnsw:construction_ef": 200,
                        "hnsw:M": 16
                    }
                )
                logger.info(f"Created new ChromaDB collection: {settings.chroma_collection_name}")
                self._add_sample_documents()
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _add_sample_documents(self) -> None:
        """Add sample documents to empty collection."""
        sample_docs = [
            "To reset your password, click on 'Forgot Password' on the login page.",
            "Our refund policy allows returns within 30 days of purchase.",
            "Customer support is available 24/7 via chat or email.",
            "To track your order, use the tracking number in your confirmation email.",
            "Account verification requires a valid email address and phone number."
        ]
        
        try:
            chunks = sample_docs
            embeddings = self.embed_documents(chunks)
            metadatas = [{"type": "sample", "category": "faq"} for _ in chunks]
            ids = [hashlib.md5(doc.encode()).hexdigest() for doc in chunks]
            
            self.collection.add(
                documents=chunks,
                embeddings=[emb.tolist() for emb in embeddings],
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(sample_docs)} sample documents")
        except Exception as e:
            logger.warning(f"Failed to add sample documents: {e}")
    
    def _prepare_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]],
        ids: Optional[List[str]],
        chunk: bool
    ) -> Dict[str, Any]:
        """Prepare documents for indexing."""
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for idx, doc in enumerate(documents):
            if chunk and len(doc.split()) > CHUNK_SIZE:
                chunks = self.chunk_document(doc)
                for chunk_idx, (chunk_text, chunk_meta) in enumerate(chunks):
                    all_chunks.append(chunk_text)
                    
                    combined_meta = chunk_meta.copy()
                    if metadatas and idx < len(metadatas):
                        combined_meta.update(metadatas[idx])
                    combined_meta['doc_index'] = idx
                    all_metadatas.append(combined_meta)
                    
                    if ids and idx < len(ids):
                        chunk_id = f"{ids[idx]}_chunk_{chunk_idx}"
                    else:
                        chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()
                    all_ids.append(chunk_id)
            else:
                all_chunks.append(doc)
                
                meta = {"doc_index": idx}
                if metadatas and idx < len(metadatas):
                    meta.update(metadatas[idx])
                all_metadatas.append(meta)
                
                if ids and idx < len(ids):
                    all_ids.append(ids[idx])
                else:
                    all_ids.append(hashlib.md5(doc.encode()).hexdigest())
        
        return {
            "chunks": all_chunks,
            "metadatas": all_metadatas,
            "ids": all_ids
        }
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a search query."""
        prefixed_query = QUERY_PREFIX + query
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
        
        Version 2.0.0: FIXED - Adaptive batch sizing.
        """
        # Adaptive batch size based on document count
        if len(documents) < 10:
            batch_size = len(documents)
        elif len(documents) < 100:
            batch_size = 32
        else:
            batch_size = settings.embedding_batch_size
        
        prefixed_docs = [DOC_PREFIX + doc for doc in documents]
        embeddings = self.embedder.encode(
            prefixed_docs,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=len(documents) > 10,
            convert_to_numpy=True
        )
        return embeddings
    
    def chunk_document(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Split document into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_words = words[i:i + CHUNK_SIZE]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_words) >= CHUNK_OVERLAP:
                metadata = {
                    "chunk_index": len(chunks),
                    "start_word": i,
                    "end_word": min(i + CHUNK_SIZE, len(words)),
                    "total_words": len(words)
                }
                chunks.append((chunk_text, metadata))
        
        return chunks


__all__ = ['RAGTool']
```

Due to length constraints, I'll continue with the remaining files (Memory, Escalation, Attachment tools and __init__.py) in the next response. Should I continue?
