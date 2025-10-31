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
