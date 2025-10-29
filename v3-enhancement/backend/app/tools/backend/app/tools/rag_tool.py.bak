"""
RAG (Retrieval-Augmented Generation) tool implementation.
Uses EmbeddingGemma for embeddings and ChromaDB for vector storage.

Phase 1 Update: Async-first interface with ToolResult return types.
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
from chromadb.utils import embedding_functions

from ..config import settings
from ..services.cache_service import CacheService
from .base_tool import BaseTool, ToolResult, ToolStatus

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
    
    Phase 1: Implements async-first interface with ToolResult returns.
    """
    
    def __init__(self):
        """Initialize RAG tool with embedding model and vector store."""
        # Call new-style parent init (no auto-initialization)
        super().__init__(
            name="rag_search",
            description="Search knowledge base for relevant information using semantic similarity"
        )
        
        # Resources will be initialized in async initialize()
        self.embedder = None
        self.chroma_client = None
        self.collection = None
        self.cache = None
    
    # ===========================
    # Async Interface (Phase 1)
    # ===========================
    
    async def initialize(self) -> None:
        """
        Initialize RAG tool resources (async-safe).
        Sets up embedding model, ChromaDB, and cache service.
        """
        try:
            
