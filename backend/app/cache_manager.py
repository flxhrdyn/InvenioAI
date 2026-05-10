import json
import logging
from typing import Any, Optional

import diskcache
import redis
import threading

from .config import BASE_DIR, CACHE_TYPE, REDIS_URL

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, cache_type: str = CACHE_TYPE, redis_url: str = REDIS_URL):
        self.cache_type = cache_type
        self.redis_client = None
        self.disk_cache = None
        self._semantic_lock = threading.Lock()
        
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

    def get_semantic(self, query_embedding: list[float], threshold: float = 0.95) -> Optional[str]:
        """Find a similar query in the semantic registry and return its cache key.
        
        This uses a simple linear scan of previous query embeddings. 
        For portfolio scale, this is very fast.
        """
        if self.cache_type != "diskcache" or self.disk_cache is None:
            # We currently only support semantic cache in diskcache for simplicity
            return None
            
        try:
            registry = self.disk_cache.get("semantic_registry", [])
            if not registry:
                return None
                
            import numpy as np
            query_vec = np.array(query_embedding)
            
            best_score = -1.0
            best_key = None
            
            for entry in registry:
                # entry: {"vector": [...], "key": "rag_cache:..."}
                entry_vec = np.array(entry["vector"])
                
                # Cosine similarity formula: (A . B) / (||A|| * ||B||)
                norm_a = np.linalg.norm(query_vec)
                norm_b = np.linalg.norm(entry_vec)
                if norm_a == 0 or norm_b == 0:
                    continue
                    
                score = np.dot(query_vec, entry_vec) / (norm_a * norm_b)
                
                if score > best_score:
                    best_score = score
                    best_key = entry["key"]
                    
            if best_score >= threshold:
                logger.info(f"Semantic Cache HIT: score={best_score:.4f}")
                return best_key
            else:
                logger.info(f"Semantic Cache MISS: best_score={best_score:.4f}, threshold={threshold}")
        except Exception as e:
            logger.warning(f"Semantic Cache lookup failed: {e}")
            
        return None

    def add_semantic(self, query_embedding: list[float], cache_key: str):
        """Add a new query embedding to the semantic registry."""
        if self.cache_type != "diskcache" or self.disk_cache is None:
            return
            
        try:
            # Using a thread lock to prevent concurrent registry updates in the same process.
            # DiskCache handles cross-process safety for get/set, but we need atomicity for append.
            with self._semantic_lock:
                registry = self.disk_cache.get("semantic_registry", [])
                registry.append({
                    "vector": query_embedding,
                    "key": cache_key
                })
                # Keep registry size manageable for linear scan (last 1000 items)
                if len(registry) > 1000:
                    registry.pop(0)
                self.disk_cache.set("semantic_registry", registry)
        except Exception as e:
            logger.warning(f"Failed to add to semantic registry: {e}")
