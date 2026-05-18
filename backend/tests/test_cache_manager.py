import pytest
import numpy as np
from unittest.mock import MagicMock
from app.cache_manager import CacheManager

def test_get_semantic_correctness():
    # Mock diskcache
    mock_cache = MagicMock()
    
    # Setup registry with one entry
    vec = [0.1, 0.2, 0.3]
    mock_cache.get.return_value = [{"vector": vec, "key": "hit_key"}]
    
    manager = CacheManager(cache_type="diskcache")
    manager.disk_cache = mock_cache
    
    # Test HIT
    hit_key = manager.get_semantic(vec, threshold=0.99)
    assert hit_key == "hit_key"
    
    # Test MISS
    miss_key = manager.get_semantic([0.9, 0.9, 0.9], threshold=0.99)
    assert miss_key is None

def test_get_semantic_handles_zero_norm():
    mock_cache = MagicMock()
    mock_cache.get.return_value = [{"vector": [0, 0, 0], "key": "zero_key"}]
    
    manager = CacheManager(cache_type="diskcache")
    manager.disk_cache = mock_cache
    
    # Should not crash and return None
    key = manager.get_semantic([0.1, 0.2, 0.3])
    assert key is None
