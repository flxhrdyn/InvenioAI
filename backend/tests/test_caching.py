import pytest
from unittest.mock import MagicMock, patch
from app.rag_pipeline import rag_pipeline
from app.cache_manager import CacheManager

@pytest.fixture
def mock_cache():
    with patch("app.rag_pipeline.get_cache_manager") as mock:
        cache_inst = MagicMock(spec=CacheManager)
        mock.return_value = cache_inst
        yield cache_inst

def test_rag_pipeline_uses_cache(mock_cache):
    """Test that rag_pipeline checks and uses the cache."""
    # Setup cache hit
    cached_result = {"answer": "Cached answer", "sources": []}
    mock_cache.get.return_value = cached_result
    
    question = "What is AI?"
    history = []
    
    # Execute
    result = rag_pipeline(question, history)
    
    # Assert
    assert result == cached_result
    mock_cache.get.assert_called_once()
    
def test_rag_pipeline_saves_to_cache_on_miss(mock_cache):
    """Test that rag_pipeline saves to cache when it's a miss."""
    # Setup cache miss
    mock_cache.get.return_value = None
    
    from langchain_core.documents import Document
    
    with patch("app.rag_pipeline._run_rag_pipeline_once") as mock_run:
        real_result = {"answer": "Real answer", "sources": []}
        mock_run.return_value = real_result
        
        question = "New question"
        history = []
        
        # Execute
        result = rag_pipeline(question, history)
        
        # Assert
        assert result == real_result
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_called_once()

@pytest.mark.asyncio
async def test_rag_pipeline_stream_async_uses_cache(mock_cache):
    """Test that rag_pipeline_stream_async checks and uses the cache."""
    from app.rag_pipeline import rag_pipeline_stream_async
    import json
    
    # Setup cache hit
    cached_result = {"answer": "Cached async answer", "sources": []}
    mock_cache.get.return_value = cached_result
    
    query = "Stream question"
    history = []
    
    # Execute
    chunks = []
    async for chunk in rag_pipeline_stream_async(query, history):
        chunks.append(json.loads(chunk.strip()))
        
    # Assert
    assert any(c["step"] == "cached" for c in chunks)
    assert any(c["step"] == "done" and c["answer"] == "Cached async answer" for c in chunks)
    mock_cache.get.assert_called_once()
