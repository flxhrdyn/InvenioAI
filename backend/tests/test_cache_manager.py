import pytest
from app.cache_manager import CacheManager
from unittest.mock import patch, MagicMock

def test_cache_set_and_get():
    # Setup
    # We will force it to use diskcache for predictable local testing
    cache = CacheManager(cache_type="diskcache")
    
    # Execution
    cache.set("test_key", "test_value")
    result = cache.get("test_key")
    
    # Assertions
    assert result == "test_value"

def test_cache_miss_returns_none():
    cache = CacheManager(cache_type="diskcache")
    result = cache.get("nonexistent_key")
    assert result is None

@patch("app.cache_manager.redis.Redis.from_url")
def test_redis_backend_initialization(mock_redis):
    # Setup
    mock_redis_instance = MagicMock()
    mock_redis.return_value = mock_redis_instance
    
    cache = CacheManager(cache_type="redis", redis_url="redis://dummy")
    
    # Execution
    cache.set("k", "v")
    
    # Assertions
    assert mock_redis.called
    mock_redis_instance.setex.assert_called_once()
