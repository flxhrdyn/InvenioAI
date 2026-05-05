import pytest
import json
import time
from unittest.mock import patch, MagicMock
from app.rag_pipeline import rag_pipeline, rag_pipeline_stream_async
from app.metrics import load_metrics, reset_metrics

@pytest.fixture(autouse=True)
def setup_metrics():
    reset_metrics()
    yield

def test_rag_pipeline_logs_metrics_on_full_run():
    print("Starting full run test...")
    # Mock lower level components to exercise the real _run_rag_pipeline_with_query
    with patch("app.rag_pipeline.build_retriever") as mock_build, \
         patch("app.rag_pipeline.retrieve_documents") as mock_retrieve, \
         patch("app.rag_pipeline.rerank") as mock_rerank, \
         patch("app.rag_pipeline._get_llm") as mock_llm, \
         patch("app.rag_pipeline.rewrite_query", return_value="standalone"), \
         patch("app.rag_pipeline.get_cache_manager") as mock_cache:
        
        # Setup mocks
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_relevance_scores.return_value = []
        
        mock_build.return_value = (MagicMock(), mock_vectorstore, MagicMock())
        mock_retrieve.return_value = ([MagicMock(page_content="doc1")], {"mode": "dense"})
        mock_rerank.return_value = [MagicMock(page_content="doc1", metadata={"source": "test.pdf"})]
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = MagicMock(content="Real-ish Answer")
        mock_llm.return_value = mock_llm_instance
        
        # Ensure cache miss
        mock_cache.return_value.get.return_value = None
        
        rag_pipeline("Test Question", [])
        
        metrics = load_metrics()
        assert metrics["total_queries"] == 1
        assert len(metrics["query_history"]) == 1
        assert metrics["query_history"][0]["question"] == "Test Question"
        assert metrics["query_history"][0]["docs_retrieved"] == 1

def test_rag_pipeline_logs_metrics_on_cache_hit():
    cached_data = {
        "answer": "Cached Answer",
        "sources": "[]",
        "metrics": {
            "total_time": 0.5,
            "docs_retrieved": 10,
            "chunks_processed": 4,
            "retrieval_scores": [0.95]
        }
    }
    
    with patch("app.rag_pipeline.get_cache_manager") as mock_cache:
        # Layer 1 hit
        mock_cache.return_value.get.return_value = cached_data
        
        rag_pipeline("Test Question", [])
        
        metrics = load_metrics()
        assert metrics["total_queries"] == 1
        assert metrics["query_history"][0]["question"] == "Test Question"
        assert metrics["query_history"][0]["response_time"] == 0.01 # Should be fixed cache hit time
        assert metrics["query_history"][0]["docs_retrieved"] == 10

@pytest.mark.asyncio
async def test_rag_pipeline_stream_async_logs_metrics():
    # Mock the async generator parts
    with patch("app.rag_pipeline.rewrite_query_async", return_value="standalone"), \
         patch("app.rag_pipeline.build_retriever") as mock_retriever_build, \
         patch("app.rag_pipeline.retrieve_documents_async") as mock_retrieve, \
         patch("app.rag_pipeline.rerank", return_value=[MagicMock(page_content="doc1")]), \
         patch("app.rag_pipeline._get_llm") as mock_llm, \
         patch("app.rag_pipeline.get_cache_manager") as mock_cache:
        
        # Mocking async retriever build
        mock_retriever_build.return_value = (MagicMock(), MagicMock(), MagicMock())
        
        # Mocking async retrieval
        mock_retrieve.return_value = ([MagicMock(page_content="doc1")], {"mode": "dense", "retrieval_scores": [0.9]})
        
        # Mocking LLM stream correctly
        async def mock_astream(*args, **kwargs):
            yield MagicMock(content="Mocked")
            yield MagicMock(content=" Answer")
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.astream = mock_astream
        mock_llm.return_value = mock_llm_instance
        
        # Ensure cache miss
        mock_cache.return_value.get.return_value = None
        
        # Run the async generator
        results = []
        async for chunk in rag_pipeline_stream_async("Test Async", []):
            results.append(chunk)
            
        assert len(results) > 0
        metrics = load_metrics()
        assert metrics["total_queries"] == 1
        assert metrics["query_history"][0]["question"] == "Test Async"
