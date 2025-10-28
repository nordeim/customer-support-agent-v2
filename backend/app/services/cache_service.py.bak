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
