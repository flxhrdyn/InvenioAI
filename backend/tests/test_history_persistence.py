import pytest
from app.cache_manager import CacheManager

def test_cache_manager_stores_chat_history():
    """Verify that CacheManager can store and retrieve a list of chat messages."""
    cache = CacheManager()
    history_key = "test_chat_history"
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!", "sources": []}
    ]
    
    # Action
    cache.set(history_key, messages)
    retrieved = cache.get(history_key)
    
    # Assert
    assert retrieved == messages
    assert isinstance(retrieved, list)
    assert retrieved[0]["role"] == "user"

def test_cache_manager_clears_history():
    """Verify that setting history to None or empty effectively clears it."""
    cache = CacheManager()
    history_key = "test_chat_history_clear"
    cache.set(history_key, [{"role": "user", "content": "test"}])
    
    # Action
    cache.set(history_key, None)
    retrieved = cache.get(history_key)
    
    # Assert
    assert retrieved is None
