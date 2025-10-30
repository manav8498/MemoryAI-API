"""
Redis cache connection and utilities.
Provides caching layer for API responses and temporary data storage.
"""
import json
from typing import Any, Optional
import redis.asyncio as aioredis
from redis.asyncio import Redis

from backend.core.config import settings
from backend.core.logging_config import logger


# Global Redis client instance
redis_client: Optional[Redis] = None


async def get_redis() -> Redis:
    """
    Get Redis client instance.
    Creates connection if it doesn't exist.
    """
    global redis_client

    if redis_client is None:
        redis_client = await aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=50,
        )

    return redis_client


async def close_redis() -> None:
    """
    Close Redis connection.
    Called during application shutdown.
    """
    global redis_client

    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")


async def check_redis_connection() -> bool:
    """
    Check if Redis connection is healthy.
    Used for readiness checks.
    """
    try:
        client = await get_redis()
        await client.ping()
        return True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False


class CacheManager:
    """
    High-level cache manager with common operations.
    """

    def __init__(self, redis: Redis):
        self.redis = redis

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time to live in seconds (defaults to settings.CACHE_TTL_SECONDS)
        """
        try:
            ttl = ttl or settings.CACHE_TTL_SECONDS
            serialized = json.dumps(value)
            await self.redis.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.

        Args:
            pattern: Redis pattern (e.g., "user:*", "memory:123:*")

        Returns:
            Number of keys deleted
        """
        try:
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                return await self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache delete pattern error for {pattern}: {e}")
            return 0

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter."""
        try:
            return await self.redis.incrby(key, amount)
        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            return 0

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from cache."""
        try:
            if not keys:
                return {}

            values = await self.redis.mget(keys)
            result = {}

            for key, value in zip(keys, values):
                if value:
                    try:
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        result[key] = value

            return result
        except Exception as e:
            logger.error(f"Cache get_many error: {e}")
            return {}

    async def set_many(
        self,
        items: dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """Set multiple values in cache."""
        try:
            ttl = ttl or settings.CACHE_TTL_SECONDS
            pipe = self.redis.pipeline()

            for key, value in items.items():
                serialized = json.dumps(value)
                pipe.setex(key, ttl, serialized)

            await pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Cache set_many error: {e}")
            return False


async def get_cache_manager() -> CacheManager:
    """Dependency for getting cache manager."""
    redis = await get_redis()
    return CacheManager(redis)
