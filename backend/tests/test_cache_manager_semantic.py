import pytest
import numpy as np
from app.cache_manager import CacheManager
import tempfile
import shutil
from pathlib import Path

@pytest.fixture
def temp_cache():
    """Create a temporary CacheManager with diskcache."""
    temp_dir = tempfile.mkdtemp()
    with patch("app.cache_manager.BASE_DIR", Path(temp_dir)):
        cache = CacheManager(cache_type="diskcache")
        yield cache
        # Explicitly close the cache to release file handles on Windows
        if cache.disk_cache:
            cache.disk_cache.close()
            
    # Best-effort cleanup for Windows
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass

from unittest.mock import patch

def test_get_semantic_empty(temp_cache):
    """Verify get_semantic returns None when registry is empty."""
    # We need to provide a dummy embedding (list of floats)
    embedding = [0.1] * 384
    result = temp_cache.get_semantic(embedding)
    assert result is None

def test_add_and_get_semantic_hit(temp_cache):
    """Verify semantic cache hit for highly similar vectors."""
    # Base vector
    vec1 = [1.0, 0.0, 0.0] + ([0.0] * 381)
    key1 = "result_key_1"
    
    temp_cache.add_semantic(vec1, key1)
    
    # Highly similar vector (cosine similarity ~0.999)
    vec2 = [0.999, 0.044, 0.0] + ([0.0] * 381)
    
    result = temp_cache.get_semantic(vec2, threshold=0.95)
    assert result == key1

def test_add_and_get_semantic_miss(temp_cache):
    """Verify semantic cache miss for dissimilar vectors."""
    vec1 = [1.0, 0.0, 0.0] + ([0.0] * 381)
    key1 = "result_key_1"
    
    temp_cache.add_semantic(vec1, key1)
    
    # Dissimilar vector (orthogonal)
    vec2 = [0.0, 1.0, 0.0] + ([0.0] * 381)
    
    result = temp_cache.get_semantic(vec2, threshold=0.95)
    assert result is None

def test_registry_limit(temp_cache):
    """Verify semantic registry size is capped at 1000."""
    from unittest.mock import MagicMock
    
    mock_dict = {}
    
    def mock_get(key, default=None):
        return mock_dict.get(key, default)
        
    def mock_set(key, value):
        mock_dict[key] = value
        
    temp_cache.disk_cache = MagicMock()
    temp_cache.disk_cache.get.side_effect = mock_get
    temp_cache.disk_cache.set.side_effect = mock_set
    
    for i in range(1100):
        vec = [float(i)]
        temp_cache.add_semantic(vec, f"key_{i}")
        
    registry = mock_dict.get("semantic_registry")
    assert len(registry) == 1000
    # Should keep the LATEST ones
    assert registry[-1]["key"] == "key_1099"
    assert registry[0]["key"] == "key_100"
