import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from app.main import app
from langchain_core.documents import Document

client = TestClient(app)

@patch("app.rag_pipeline.build_retriever")
@patch("app.rag_pipeline._get_llm")
@patch("app.rag_pipeline.rewrite_query")
@patch("app.rag_pipeline.retrieve_documents")
@patch("app.rag_pipeline.rerank")
def test_query_returns_source_nodes_with_metadata(mock_rerank, mock_retrieve, mock_rewrite, mock_llm, mock_build):
    # Setup mocks
    mock_build.return_value = (MagicMock(), MagicMock(), MagicMock())
    mock_rewrite.return_value = "standalone query"
    mock_llm.return_value.invoke.return_value.content = "Ini jawaban AI."
    
    # Mock documents
    docs = [
        Document(page_content="Snippet teks 1", metadata={"source": "file_a.pdf", "score": 0.9}),
        Document(page_content="Snippet teks 2", metadata={"source": "file_b.pdf", "score": 0.8})
    ]
    mock_retrieve.return_value = (docs, {"mode": "dense"})
    mock_rerank.return_value = docs
    
    # Execute
    response = client.post("/query", json={"question": "test query"})
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    
    assert "sources" in data
    assert len(data["sources"]) == 2
    
    # Verify first source structure
    s1 = data["sources"][0]
    assert s1["file"] == "file_a.pdf"
    assert s1["text"] == "Snippet teks 1"
    assert "score" in s1
