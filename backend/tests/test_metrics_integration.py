import pytest
import os
import json
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from app.main import app
from app.config import METRICS_FILE

client = TestClient(app)

def test_query_updates_metrics():
    # Ensure metrics file is reset or doesn't exist
    if os.path.exists(METRICS_FILE):
        os.remove(METRICS_FILE)
    
    from langchain_core.documents import Document
    
    with patch("app.rag_pipeline.build_retriever") as mock_build, \
         patch("app.rag_pipeline.retrieve_documents") as mock_retrieve, \
         patch("app.rag_pipeline.rerank") as mock_rerank, \
         patch("app.rag_pipeline._get_llm") as mock_llm:
        
        # Setup mocks
        mock_build.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_retrieve.return_value = ([Document(page_content="test", metadata={"source": "test.pdf"})], {"mode": "dense"})
        mock_rerank.return_value = [Document(page_content="test", metadata={"source": "test.pdf"})]
        
        mock_llm_inst = MagicMock()
        mock_llm_inst.invoke.return_value = MagicMock(content="AI is cool")
        mock_llm.return_value = mock_llm_inst
        
        response = client.post("/query", json={"question": "What is AI?", "history": []})
        
        assert response.status_code == 200
        
        # Now check if metrics.json was created and updated
        assert os.path.exists(METRICS_FILE)
        
        with open(METRICS_FILE, "r") as f:
            metrics = json.load(f)
            assert metrics["total_queries"] >= 1
            assert len(metrics["query_history"]) >= 1
            # Check if the last entry is our question
            assert metrics["query_history"][-1]["question"] == "What is AI?"
            assert "response_time" in metrics["query_history"][-1]
