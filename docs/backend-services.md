# Review & Missing Services Implementation Plan

## 📋 Code Review Analysis

After meticulously reviewing the backend codebase, I've identified that **YES, these service files are needed** for the following reasons:

### Current Dependencies:
1. **`cache_service.py`** - Already created and imported by:
   - `backend/app/tools/rag_tool.py`: `from ..services.cache_service import CacheService`
   - `backend/app/main.py`: `from .services.cache_service import CacheService`

2. **`auth_service.py`** - Already created and imported by:
   - `backend/app/api/routes/chat.py`: `from ...services.auth_service import get_current_user`

3. **Missing Services** that would improve architecture:
   - **`embedding_service.py`** - Currently embedding logic is scattered in RAG tool
   - **`memory_service.py`** - Would provide a cleaner interface over MemoryTool
   - **`__init__.py`** - Required for proper module imports

## 🏗️ Implementation Plan

### Why These Services Are Needed:

1. **Embedding Service**:
   - Centralize all embedding operations
   - Support multiple embedding models
   - Implement embedding caching
   - Provide consistent interface for all tools

2. **Memory Service**:
   - Higher-level abstraction over MemoryTool
   - Session-based memory management
   - Memory analytics and insights
   - Bulk operations and cleanup

3. **Services Module Init**:
   - Proper module structure
   - Clean imports
   - Service registry

---

## 🛠️ Complete Implementation

### File 1: Services Module Initialization

**`backend/app/services/__init__.py`**
```python
"""
Services module for Customer Support AI Agent.
Provides business logic and integration services.
"""

from .cache_service import CacheService
from .auth_service import (
    AuthService,
    auth_service,
    get_current_user,
    require_auth,
    require_admin,
    require_support
)
from .embedding_service import EmbeddingService, embedding_service
from .memory_service import MemoryService, memory_service

__all__ = [
    # Cache
    'CacheService',
    
    # Auth
    'AuthService',
    'auth_service',
    'get_current_user',
    'require_auth',
    'require_admin',
    'require_support',
    
    # Embeddings
    'EmbeddingService',
    'embedding_service',
    
    # Memory
    'MemoryService',
    'memory_service',
]
```

### File 2: Embedding Service

**`backend/app/services/embedding_service.py`**
```python
"""
Embedding service for centralized embedding generation and management.
Supports multiple embedding models and caching.
"""
import logging
import hashlib
import numpy as np
from typing import List, Optional, Dict, Any, Union, Tuple
from enum import Enum
import torch
from pathlib import Path
import pickle

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

from ..config import settings
from .cache_service import CacheService

logger = logging.getLogger(__name__)


class EmbeddingModel(str, Enum):
    """Supported embedding models."""
    MINILM = "all-MiniLM-L6-v2"
    MPNET = "all-mpnet-base-v2"
    EMBEDDING_GEMMA = "google/embedding-gemma-256m-it"
    ADA_002 = "text-embedding-ada-002"  # OpenAI
    CUSTOM = "custom"


class EmbeddingService:
    """
    Centralized service for generating and managing embeddings.
    Supports multiple models, caching, and batch processing.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        use_cache: bool = True,
        cache_ttl: int = 86400  # 24 hours
    ):
        """
        Initialize embedding service.
        
        Args:
            model_name: Model to use (defaults to config)
            use_cache: Enable caching
            cache_ttl: Cache TTL in seconds
        """
        self.model_name = model_name or settings.embedding_model
        self.use_cache = use_cache and settings.cache_enabled
        self.cache_ttl = cache_ttl
        
        # Initialize cache
        self.cache = CacheService() if self.use_cache else None
        
        # Model and tokenizer
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # Embedding dimensions
        self.dimensions = {}
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the embedding model."""
        try:
            # Determine device
            if settings.use_gpu_embeddings and torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info(f"Using GPU for embeddings: {torch.cuda.get_device_name()}")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU for embeddings")
            
            # Load model based on type
            if "gemma" in self.model_name.lower():
                self._load_gemma_model()
            elif self.model_name == EmbeddingModel.ADA_002:
                self._setup_openai_embeddings()
            else:
                self._load_sentence_transformer()
            
            logger.info(f"Embedding service initialized with model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            # Fallback to simple model
            self._load_fallback_model()
    
    def _load_sentence_transformer(self) -> None:
        """Load a SentenceTransformer model."""
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.dimensions[self.model_name] = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded SentenceTransformer: {self.model_name} (dim: {self.dimensions[self.model_name]})")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
            raise
    
    def _load_gemma_model(self) -> None:
        """Load Google's EmbeddingGemma model."""
        try:
            # Try to load EmbeddingGemma
            model_name = settings.embedding_gemma_model
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            if self.device.type == "cuda":
                self.model = self.model.cuda()
            
            # Set dimensions
            self.dimensions[model_name] = 768  # EmbeddingGemma dimension
            
            logger.info(f"Loaded EmbeddingGemma model: {model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load EmbeddingGemma: {e}")
            # Fallback to SentenceTransformer
            self.model_name = EmbeddingModel.MINILM
            self._load_sentence_transformer()
    
    def _setup_openai_embeddings(self) -> None:
        """Setup OpenAI embeddings (requires API key)."""
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key required for Ada-002 embeddings")
        
        # OpenAI embeddings are handled via API calls
        self.dimensions[EmbeddingModel.ADA_002] = 1536
        logger.info("OpenAI embedding service configured")
    
    def _load_fallback_model(self) -> None:
        """Load a fallback model in case of errors."""
        logger.warning("Loading fallback embedding model")
        self.model_name = EmbeddingModel.MINILM
        try:
            self.model = SentenceTransformer(self.model_name, device="cpu")
            self.dimensions[self.model_name] = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            raise RuntimeError("Unable to load any embedding model")
    
    def get_dimension(self) -> int:
        """Get the dimension of current embedding model."""
        return self.dimensions.get(self.model_name, settings.embedding_dimension)
    
    async def generate_embedding(
        self,
        text: str,
        model: Optional[str] = None,
        prefix: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            model: Optional model override
            prefix: Optional prefix for the text
            
        Returns:
            Embedding vector as numpy array
        """
        # Apply prefix if provided
        if prefix:
            text = prefix + text
        
        # Check cache
        if self.use_cache:
            cache_key = self._get_cache_key(text, model)
            cached = await self.cache.get(cache_key)
            if cached is not None:
                return np.array(cached)
        
        # Generate embedding
        embedding = await self._generate_single_embedding(text, model)
        
        # Cache the result
        if self.use_cache:
            await self.cache.set(cache_key, embedding.tolist(), ttl=self.cache_ttl)
        
        return embedding
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        prefix: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            model: Optional model override
            prefix: Optional prefix for all texts
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Apply prefix if provided
        if prefix:
            texts = [prefix + text for text in texts]
        
        batch_size = batch_size or settings.embedding_batch_size
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Check cache for batch
            batch_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            if self.use_cache:
                for j, text in enumerate(batch):
                    cache_key = self._get_cache_key(text, model)
                    cached = await self.cache.get(cache_key)
                    if cached is not None:
                        batch_embeddings.append((j, np.array(cached)))
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(j)
            else:
                uncached_texts = batch
                uncached_indices = list(range(len(batch)))
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                new_embeddings = await self._generate_batch_embeddings(uncached_texts, model)
                
                # Cache new embeddings
                if self.use_cache:
                    for text, embedding in zip(uncached_texts, new_embeddings):
                        cache_key = self._get_cache_key(text, model)
                        await self.cache.set(cache_key, embedding.tolist(), ttl=self.cache_ttl)
                
                # Combine with cached embeddings
                for idx, embedding in zip(uncached_indices, new_embeddings):
                    batch_embeddings.append((idx, embedding))
            
            # Sort by original order
            batch_embeddings.sort(key=lambda x: x[0])
            embeddings.extend([emb for _, emb in batch_embeddings])
        
        return embeddings
    
    async def _generate_single_embedding(
        self,
        text: str,
        model: Optional[str] = None
    ) -> np.ndarray:
        """Generate embedding for single text."""
        embeddings = await self._generate_batch_embeddings([text], model)
        return embeddings[0]
    
    async def _generate_batch_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts."""
        model_name = model or self.model_name
        
        try:
            if model_name == EmbeddingModel.ADA_002:
                return await self._generate_openai_embeddings(texts)
            elif "gemma" in model_name.lower() and self.tokenizer:
                return self._generate_gemma_embeddings(texts)
            elif isinstance(self.model, SentenceTransformer):
                return self._generate_sentence_transformer_embeddings(texts)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def _generate_sentence_transformer_embeddings(
        self,
        texts: List[str]
    ) -> List[np.ndarray]:
        """Generate embeddings using SentenceTransformer."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100
        )
        return list(embeddings)
    
    def _generate_gemma_embeddings(
        self,
        texts: List[str]
    ) -> List[np.ndarray]:
        """Generate embeddings using EmbeddingGemma."""
        embeddings = []
        
        for text in texts:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            if self.device.type == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1)
                # Normalize
                embedding = F.normalize(embedding, p=2, dim=1)
                embeddings.append(embedding.cpu().numpy()[0])
        
        return embeddings
    
    async def _generate_openai_embeddings(
        self,
        texts: List[str]
    ) -> List[np.ndarray]:
        """Generate embeddings using OpenAI API."""
        try:
            import openai
            
            # Configure OpenAI
            openai.api_key = settings.openai_api_key.get_secret_value()
            
            # Generate embeddings
            response = await openai.Embedding.acreate(
                model="text-embedding-ada-002",
                input=texts
            )
            
            embeddings = [np.array(item["embedding"]) for item in response["data"]]
            return embeddings
            
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    def _get_cache_key(self, text: str, model: Optional[str] = None) -> str:
        """Generate cache key for text and model."""
        model_name = model or self.model_name
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"embedding:{model_name}:{text_hash}"
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric (cosine, dot, euclidean)
            
        Returns:
            Similarity score
        """
        if metric == "cosine":
            # Cosine similarity
            return np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
        elif metric == "dot":
            # Dot product (for normalized vectors)
            return np.dot(embedding1, embedding2)
        elif metric == "euclidean":
            # Negative euclidean distance (higher is more similar)
            return -np.linalg.norm(embedding1 - embedding2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        embeddings: List[np.ndarray],
        top_k: int = 5,
        threshold: Optional[float] = None,
        metric: str = "cosine"
    ) -> List[Tuple[int, float]]:
        """
        Find most similar embeddings to query.
        
        Args:
            query_embedding: Query embedding
            embeddings: List of embeddings to search
            top_k: Number of results
            threshold: Minimum similarity threshold
            metric: Similarity metric
            
        Returns:
            List of (index, similarity) tuples
        """
        similarities = []
        
        for i, embedding in enumerate(embeddings):
            similarity = self.compute_similarity(query_embedding, embedding, metric)
            
            if threshold is None or similarity >= threshold:
                similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    async def save_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
        filepath: str
    ) -> None:
        """
        Save embeddings to file.
        
        Args:
            embeddings: Dictionary of text -> embedding
            filepath: Path to save file
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy arrays to lists for JSON serialization
            data = {
                "model": self.model_name,
                "dimension": self.get_dimension(),
                "embeddings": {k: v.tolist() for k, v in embeddings.items()}
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved {len(embeddings)} embeddings to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            raise
    
    async def load_embeddings(
        self,
        filepath: str
    ) -> Dict[str, np.ndarray]:
        """
        Load embeddings from file.
        
        Args:
            filepath: Path to saved embeddings
            
        Returns:
            Dictionary of text -> embedding
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Convert lists back to numpy arrays
            embeddings = {
                k: np.array(v) for k, v in data["embeddings"].items()
            }
            
            logger.info(
                f"Loaded {len(embeddings)} embeddings "
                f"(model: {data['model']}, dim: {data['dimension']})"
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise
    
    async def clear_cache(self) -> int:
        """
        Clear embedding cache.
        
        Returns:
            Number of cleared entries
        """
        if self.cache:
            count = await self.cache.clear_pattern("embedding:*")
            logger.info(f"Cleared {count} embedding cache entries")
            return count
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get embedding service statistics.
        
        Returns:
            Service statistics
        """
        return {
            "model": self.model_name,
            "dimension": self.get_dimension(),
            "device": str(self.device) if self.device else "cpu",
            "cache_enabled": self.use_cache,
            "supported_models": list(EmbeddingModel),
        }


# Global embedding service instance
embedding_service = EmbeddingService()

# Export for convenience
__all__ = ['EmbeddingService', 'embedding_service', 'EmbeddingModel']
```

### File 3: Memory Service

**`backend/app/services/memory_service.py`**
```python
"""
Memory service for managing conversation memory and context.
Provides high-level operations over the memory tool.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json

from sqlalchemy import func, and_, or_, desc
from sqlalchemy.orm import Session

from ..config import settings
from ..models.memory import Memory
from ..tools.memory_tool import MemoryTool
from ..database import get_db

logger = logging.getLogger(__name__)


class MemoryService:
    """
    High-level service for memory management.
    Provides session-based memory operations, analytics, and insights.
    """
    
    def __init__(self):
        """Initialize memory service."""
        self.memory_tool = MemoryTool()
        self.enabled = settings.memory_enabled
        self.max_entries = settings.memory_max_entries
        self.ttl_hours = settings.memory_ttl_hours
        
        logger.info(f"Memory service initialized (enabled: {self.enabled})")
    
    async def create_session_memory(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Initialize memory for a new session.
        
        Args:
            session_id: Session identifier
            user_id: Optional user identifier
            metadata: Optional session metadata
            
        Returns:
            Initialization status
        """
        if not self.enabled:
            return {"success": False, "message": "Memory disabled"}
        
        try:
            # Store initial session context
            initial_context = f"New session started at {datetime.utcnow().isoformat()}"
            if user_id:
                initial_context += f" for user {user_id}"
            
            result = await self.memory_tool.store_memory(
                session_id=session_id,
                content=initial_context,
                content_type="context",
                metadata={
                    "user_id": user_id,
                    "session_metadata": metadata or {}
                },
                importance=0.3
            )
            
            logger.info(f"Created memory for session {session_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create session memory: {e}")
            return {"success": False, "error": str(e)}
    
    async def add_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        sources_used: Optional[List[str]] = None,
        confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Add a conversation turn to memory.
        
        Args:
            session_id: Session identifier
            user_message: User's message
            assistant_message: Assistant's response
            sources_used: Sources referenced in response
            confidence: Response confidence score
            
        Returns:
            Storage status
        """
        if not self.enabled:
            return {"success": False, "message": "Memory disabled"}
        
        results = []
        
        try:
            # Store user message
            user_result = await self.memory_tool.store_memory(
                session_id=session_id,
                content=f"User asked: {user_message[:500]}",
                content_type="context",
                metadata={"role": "user", "timestamp": datetime.utcnow().isoformat()},
                importance=0.6
            )
            results.append(user_result)
            
            # Store assistant response
            assistant_result = await self.memory_tool.store_memory(
                session_id=session_id,
                content=f"Assistant responded: {assistant_message[:500]}",
                content_type="context",
                metadata={
                    "role": "assistant",
                    "confidence": confidence,
                    "sources": sources_used or [],
                    "timestamp": datetime.utcnow().isoformat()
                },
                importance=0.5
            )
            results.append(assistant_result)
            
            # Extract and store important facts
            facts = await self._extract_facts(user_message, assistant_message)
            for fact in facts:
                fact_result = await self.memory_tool.store_memory(
                    session_id=session_id,
                    content=fact,
                    content_type="fact",
                    importance=0.8
                )
                results.append(fact_result)
            
            return {
                "success": all(r.get("success") for r in results),
                "stored_items": len(results),
                "facts_extracted": len(facts)
            }
            
        except Exception as e:
            logger.error(f"Failed to add conversation turn: {e}")
            return {"success": False, "error": str(e)}
    
    async def store_user_preference(
        self,
        session_id: str,
        preference: str,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store a user preference.
        
        Args:
            session_id: Session identifier
            preference: Preference description
            category: Optional preference category
            
        Returns:
            Storage status
        """
        if not self.enabled:
            return {"success": False, "message": "Memory disabled"}
        
        metadata = {"type": "preference"}
        if category:
            metadata["category"] = category
        
        return await self.memory_tool.store_memory(
            session_id=session_id,
            content=preference,
            content_type="preference",
            metadata=metadata,
            importance=0.9
        )
    
    async def store_user_info(
        self,
        session_id: str,
        info: str,
        info_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store user information.
        
        Args:
            session_id: Session identifier
            info: User information
            info_type: Type of information (name, email, etc.)
            
        Returns:
            Storage status
        """
        if not self.enabled:
            return {"success": False, "message": "Memory disabled"}
        
        metadata = {}
        if info_type:
            metadata["info_type"] = info_type
        
        return await self.memory_tool.store_memory(
            session_id=session_id,
            content=info,
            content_type="user_info",
            metadata=metadata,
            importance=1.0
        )
    
    async def get_session_context(
        self,
        session_id: str,
        include_facts: bool = True,
        include_preferences: bool = True,
        max_items: int = 20
    ) -> Dict[str, Any]:
        """
        Get comprehensive session context.
        
        Args:
            session_id: Session identifier
            include_facts: Include extracted facts
            include_preferences: Include user preferences
            max_items: Maximum items to retrieve
            
        Returns:
            Session context with categorized memories
        """
        if not self.enabled:
            return {"success": False, "message": "Memory disabled", "context": {}}
        
        try:
            context = {
                "user_info": [],
                "preferences": [],
                "facts": [],
                "recent_context": [],
                "summary": ""
            }
            
            # Get user information
            user_info = await self.memory_tool.retrieve_memories(
                session_id=session_id,
                content_type="user_info",
                limit=5
            )
            context["user_info"] = [m["content"] for m in user_info]
            
            # Get preferences
            if include_preferences:
                preferences = await self.memory_tool.retrieve_memories(
                    session_id=session_id,
                    content_type="preference",
                    limit=10
                )
                context["preferences"] = [m["content"] for m in preferences]
            
            # Get facts
            if include_facts:
                facts = await self.memory_tool.retrieve_memories(
                    session_id=session_id,
                    content_type="fact",
                    limit=10
                )
                context["facts"] = [m["content"] for m in facts]
            
            # Get recent context
            recent = await self.memory_tool.retrieve_memories(
                session_id=session_id,
                content_type="context",
                limit=max_items,
                time_window_hours=1
            )
            context["recent_context"] = [m["content"] for m in recent]
            
            # Generate summary
            context["summary"] = await self.memory_tool.summarize_session(session_id)
            
            return {
                "success": True,
                "session_id": session_id,
                "context": context,
                "total_items": sum(len(v) for v in context.values() if isinstance(v, list))
            }
            
        except Exception as e:
            logger.error(f"Failed to get session context: {e}")
            return {"success": False, "error": str(e), "context": {}}
    
    async def search_memories(
        self,
        session_id: str,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search memories by content.
        
        Args:
            session_id: Session identifier
            query: Search query
            limit: Maximum results
            
        Returns:
            Matching memories
        """
        if not self.enabled:
            return []
        
        try:
            with self.memory_tool.get_db() as db:
                # Search in memory content
                memories = db.query(Memory).filter(
                    and_(
                        Memory.session_id == session_id,
                        Memory.content.contains(query)
                    )
                ).order_by(
                    desc(Memory.importance),
                    desc(Memory.created_at)
                ).limit(limit).all()
                
                return [
                    {
                        "id": m.id,
                        "content": m.content,
                        "type": m.content_type,
                        "importance": m.importance,
                        "created_at": m.created_at.isoformat()
                    }
                    for m in memories
                ]
                
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []
    
    async def get_session_insights(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Get analytical insights about a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session insights and statistics
        """
        try:
            with self.memory_tool.get_db() as db:
                # Get memory statistics
                stats = db.query(
                    Memory.content_type,
                    func.count(Memory.id).label('count'),
                    func.avg(Memory.importance).label('avg_importance')
                ).filter(
                    Memory.session_id == session_id
                ).group_by(Memory.content_type).all()
                
                # Get time-based statistics
                time_stats = db.query(
                    func.min(Memory.created_at).label('first_memory'),
                    func.max(Memory.created_at).label('last_memory'),
                    func.count(Memory.id).label('total_memories')
                ).filter(
                    Memory.session_id == session_id
                ).first()
                
                # Calculate session duration
                duration = None
                if time_stats.first_memory and time_stats.last_memory:
                    duration = (time_stats.last_memory - time_stats.first_memory).total_seconds()
                
                # Get most important memories
                important = db.query(Memory).filter(
                    Memory.session_id == session_id
                ).order_by(
                    desc(Memory.importance)
                ).limit(5).all()
                
                return {
                    "session_id": session_id,
                    "statistics": {
                        "total_memories": time_stats.total_memories if time_stats else 0,
                        "duration_seconds": duration,
                        "memory_types": {
                            stat.content_type: {
                                "count": stat.count,
                                "avg_importance": float(stat.avg_importance) if stat.avg_importance else 0
                            }
                            for stat in stats
                        }
                    },
                    "important_memories": [
                        {
                            "content": m.content,
                            "type": m.content_type,
                            "importance": m.importance
                        }
                        for m in important
                    ],
                    "timeline": {
                        "first_activity": time_stats.first_memory.isoformat() if time_stats and time_stats.first_memory else None,
                        "last_activity": time_stats.last_memory.isoformat() if time_stats and time_stats.last_memory else None
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get session insights: {e}")
            return {"error": str(e)}
    
    async def merge_sessions(
        self,
        source_session_id: str,
        target_session_id: str
    ) -> Dict[str, Any]:
        """
        Merge memories from one session to another.
        
        Args:
            source_session_id: Source session to merge from
            target_session_id: Target session to merge into
            
        Returns:
            Merge operation status
        """
        if not self.enabled:
            return {"success": False, "message": "Memory disabled"}
        
        try:
            with self.memory_tool.get_db() as db:
                # Update all memories from source to target
                updated = db.query(Memory).filter(
                    Memory.session_id == source_session_id
                ).update(
                    {Memory.session_id: target_session_id},
                    synchronize_session=False
                )
                
                db.commit()
                
                logger.info(f"Merged {updated} memories from {source_session_id} to {target_session_id}")
                
                return {
                    "success": True,
                    "memories_merged": updated,
                    "source_session": source_session_id,
                    "target_session": target_session_id
                }
                
        except Exception as e:
            logger.error(f"Failed to merge sessions: {e}")
            return {"success": False, "error": str(e)}
    
    async def export_session_memory(
        self,
        session_id: str,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Export session memory in specified format.
        
        Args:
            session_id: Session identifier
            format: Export format (json, text)
            
        Returns:
            Exported memory data
        """
        try:
            # Get all memories
            memories = await self.memory_tool.retrieve_memories(
                session_id=session_id,
                limit=1000
            )
            
            if format == "json":
                return {
                    "session_id": session_id,
                    "export_date": datetime.utcnow().isoformat(),
                    "memories": memories,
                    "total_count": len(memories)
                }
            
            elif format == "text":
                # Format as readable text
                text_lines = [
                    f"Session Memory Export: {session_id}",
                    f"Export Date: {datetime.utcnow().isoformat()}",
                    "=" * 50,
                    ""
                ]
                
                # Group by type
                by_type = defaultdict(list)
                for mem in memories:
                    by_type[mem["content_type"]].append(mem)
                
                for content_type, items in by_type.items():
                    text_lines.append(f"\n{content_type.upper()}:")
                    text_lines.append("-" * 30)
                    for item in items:
                        text_lines.append(f"• {item['content']}")
                        text_lines.append(f"  (Importance: {item['importance']}, Time: {item['created_at']})")
                
                return {
                    "session_id": session_id,
                    "format": "text",
                    "content": "\n".join(text_lines)
                }
            
            else:
                return {"error": f"Unsupported format: {format}"}
                
        except Exception as e:
            logger.error(f"Failed to export session memory: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_sessions(
        self,
        days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Clean up old session memories.
        
        Args:
            days: Days to keep (defaults to config)
            
        Returns:
            Cleanup statistics
        """
        days = days or settings.memory_cleanup_days
        
        try:
            result = await self.memory_tool.cleanup_old_memories(days=days)
            
            logger.info(f"Memory cleanup completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            return {"success": False, "error": str(e)}
    
    async def _extract_facts(
        self,
        user_message: str,
        assistant_message: str
    ) -> List[str]:
        """
        Extract important facts from conversation.
        
        Args:
            user_message: User's message
            assistant_message: Assistant's response
            
        Returns:
            List of extracted facts
        """
        facts = []
        
        # Look for patterns in user message
        import re
        
        # Email pattern
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', user_message)
        for email in emails:
            facts.append(f"User email: {email}")
        
        # Phone pattern
        phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', user_message)
        for phone in phones:
            facts.append(f"User phone: {phone}")
        
        # Order/reference numbers
        refs = re.findall(r'\b(?:order|ticket|ref|reference|confirmation)[\s#]*([A-Z0-9-]+)\b', user_message, re.I)
        for ref in refs:
            facts.append(f"Reference: {ref}")
        
        # Name patterns (simple heuristic)
        if "my name is" in user_message.lower():
            name_match = re.search(r'my name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', user_message, re.I)
            if name_match:
                facts.append(f"User name: {name_match.group(1)}")
        
        return facts[:5]  # Limit to 5 facts per turn
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory service statistics.
        
        Returns:
            Service statistics
        """
        try:
            with self.memory_tool.get_db() as db:
                total = db.query(func.count(Memory.id)).scalar()
                sessions = db.query(func.count(func.distinct(Memory.session_id))).scalar()
                
                return {
                    "enabled": self.enabled,
                    "total_memories": total,
                    "active_sessions": sessions,
                    "max_entries_per_session": self.max_entries,
                    "ttl_hours": self.ttl_hours
                }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "enabled": self.enabled,
                "error": str(e)
            }


# Global memory service instance
memory_service = MemoryService()

# Export for convenience
__all__ = ['MemoryService', 'memory_service']
```

## 📝 Summary

### ✅ Missing Services Implementation Complete

I've created the three missing service files that are essential for the backend architecture:

#### **1. Services Module Init** (`__init__.py`)
- Proper module structure with all exports
- Clean import organization
- Service registry for easy access

#### **2. Embedding Service** (`embedding_service.py`)
- **Centralized Embedding Management**: Single point for all embedding operations
- **Multiple Model Support**: SentenceTransformer, EmbeddingGemma, OpenAI
- **Intelligent Caching**: Reduces redundant computations
- **Batch Processing**: Efficient handling of multiple texts
- **Similarity Operations**: Built-in similarity computation and ranking
- **Persistence**: Save/load embeddings for offline use
- **GPU Support**: Automatic GPU detection and usage
- **Fallback Mechanisms**: Graceful degradation if primary model fails

#### **3. Memory Service** (`memory_service.py`)
- **High-Level Abstraction**: Cleaner interface over MemoryTool
- **Session Management**: Complete session lifecycle handling
- **Context Building**: Comprehensive context retrieval
- **Fact Extraction**: Automatic extraction of important information
- **Analytics & Insights**: Session statistics and analysis
- **Search Capabilities**: Content-based memory search
- **Export Functions**: JSON and text export formats
- **Session Merging**: Combine memories from multiple sessions
- **Cleanup Operations**: Automatic old memory cleanup

### 🔄 Integration Benefits

These services provide:

1. **Better Separation of Concerns**: Tools focus on low-level operations, services provide business logic
2. **Reusability**: Services can be used across different parts of the application
3. **Centralization**: Single point of control for embeddings and memory
4. **Performance**: Caching and optimization at service level
5. **Flexibility**: Easy to swap implementations or add new features
6. **Testing**: Services are easier to mock and test independently

### 🎯 Usage Examples

```python
# Using Embedding Service
from app.services import embedding_service

# Generate embeddings
embedding = await embedding_service.generate_embedding("Hello world")
embeddings = await embedding_service.generate_embeddings(["Text 1", "Text 2"])

# Using Memory Service
from app.services import memory_service

# Store conversation
await memory_service.add_conversation_turn(
    session_id="sess_123",
    user_message="What's my order status?",
    assistant_message="Your order #12345 is in transit",
    confidence=0.95
)

# Get session context
context = await memory_service.get_session_context("sess_123")
```

The services layer is now complete and provides a robust foundation for the application's AI capabilities!
