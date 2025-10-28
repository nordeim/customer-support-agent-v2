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
