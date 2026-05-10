import pytest
import json
from unittest.mock import MagicMock, patch
from app.rag_pipeline import rag_pipeline, rag_pipeline_stream_async
from app.cache_manager import CacheManager

@pytest.fixture
def mock_cache_and_embeds():
    with patch("app.rag_pipeline.get_cache_manager") as mock_cache_mgr, \
         patch("app.rag_pipeline.get_embeddings") as mock_embeds:
        
        cache_inst = MagicMock(spec=CacheManager)
        mock_cache_mgr.return_value = cache_inst
        
        embedder_inst = MagicMock()
        mock_embeds.return_value = embedder_inst
        
        yield cache_inst, embedder_inst

def test_rag_pipeline_semantic_hit(mock_cache_and_embeds):
    """Verify rag_pipeline uses semantic cache when quick/deep cache miss but semantic hit."""
    cache, embedder = mock_cache_and_embeds
    
    # 1. Quick cache miss
    # 2. Deep cache miss
    cache.get.side_effect = [None, None, {"answer": "Semantic hit answer", "sources": []}]
    
    # Semantic match hit
    cache.get_semantic.return_value = "semantic_deep_key"
    
    with patch("app.rag_pipeline.rewrite_query", return_value="standalone query"), \
         patch("app.rag_pipeline._run_rag_pipeline_with_query") as mock_run:
        
        result = rag_pipeline("Same question?", [])
        
        assert result["answer"] == "Semantic hit answer"
        assert cache.get_semantic.called
        # Verify it skipped the real RAG run
        mock_run.assert_not_called()

def test_rag_pipeline_semantic_miss_saves(mock_cache_and_embeds):
    """Verify rag_pipeline saves to semantic cache on miss."""
    cache, embedder = mock_cache_and_embeds
    
    # Miss everything
    cache.get.return_value = None
    cache.get_semantic.return_value = None
    
    real_result = {"answer": "Fresh answer", "sources": []}
    
    with patch("app.rag_pipeline.rewrite_query", return_value="standalone query"), \
         patch("app.rag_pipeline._run_rag_pipeline_with_query", return_value=real_result):
        
        result = rag_pipeline("New question?", [])
        
        assert result == real_result
        assert cache.add_semantic.called

@pytest.mark.asyncio
async def test_rag_pipeline_stream_async_semantic_hit(mock_cache_and_embeds):
    """Verify async streaming pipeline uses semantic cache."""
    cache, embedder = mock_cache_and_embeds
    
    # Quick cache miss, then Deep cache miss, then Semantic hit data retrieval
    cache.get.side_effect = [None, None, {"answer": "Async semantic hit", "sources": []}]
    cache.get_semantic.return_value = "sem_key"
    
    with patch("app.rag_pipeline.rewrite_query_async", return_value="standalone async"):
        
        chunks = []
        async for chunk in rag_pipeline_stream_async("Async question?", []):
            chunks.append(json.loads(chunk.strip()))
            
        assert any(c["step"] == "cached" for c in chunks)
        assert any(c["step"] == "done" and c["answer"] == "Async semantic hit" for c in chunks)
