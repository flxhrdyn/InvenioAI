import json
import logging
from typing import Any, Optional

import diskcache
import redis

from .config import BASE_DIR, CACHE_TYPE, REDIS_URL

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, cache_type: str = CACHE_TYPE, redis_url: str = REDIS_URL):
        self.cache_type = cache_type
        self.redis_client = None
        self.disk_cache = None
        
        if self.cache_type == "redis":
            try:
                self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
                # Force connection check immediately
                self.redis_client.ping()
                logger.info(f"Initialized Redis cache at {redis_url}")
            except Exception as e:
                logger.error(f"Failed to initialize Redis cache: {e}. Falling back to diskcache.")
                self.cache_type = "diskcache"
                self.redis_client = None
                
        if self.cache_type == "diskcache":
            cache_dir = BASE_DIR / ".cache" / "invenio"
            self.disk_cache = diskcache.Cache(cache_dir)
            logger.info(f"Initialized diskcache at {cache_dir}")

    def get(self, key: str) -> Optional[Any]:
        try:
            if self.cache_type == "redis" and self.redis_client:
                val = self.redis_client.get(key)
                if val:
                    return json.loads(val)
                return None
            elif self.cache_type == "diskcache" and self.disk_cache is not None:
                return self.disk_cache.get(key)
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
        return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        try:
            if self.cache_type == "redis" and self.redis_client:
                # We store objects as JSON strings in Redis
                self.redis_client.setex(key, ttl, json.dumps(value))
            elif self.cache_type == "diskcache" and self.disk_cache is not None:
                self.disk_cache.set(key, value, expire=ttl)
        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
