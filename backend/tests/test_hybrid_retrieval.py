from unittest.mock import MagicMock
import pytest
from app.retriever import retrieve_documents
from langchain_core.documents import Document

def test_hybrid_retrieval_returns_results():
    # Mocking components to avoid real API calls and environment requirements
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="test content", metadata={"source": "test.pdf"})
    ]
    
    mock_client = MagicMock()
    
    query = "test"
    # We test the wrapper function's ability to handle the retriever output
    docs, meta = retrieve_documents(query, dense_retriever=mock_retriever, client=mock_client)
    
    assert len(docs) > 0
    assert "mode" in meta
    assert meta["count"] == len(docs)
