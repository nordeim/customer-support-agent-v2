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
