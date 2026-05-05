import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.documents import Document
from app.retriever import retrieve_documents_async

@pytest.mark.asyncio
async def test_retrieve_documents_async_returns_correct_types_and_metadata():
    # Setup Mocks
    mock_dense_retriever = AsyncMock()
    mock_dense_retriever.ainvoke.return_value = [Document(page_content="test doc")]
    
    mock_client = MagicMock()
    # Mocking QdrantClient's scroll method for BM25 fallback check
    mock_client.scroll.return_value = ([], None)
    
    # Execute
    query = "test query"
    docs, metadata = await retrieve_documents_async(query, mock_dense_retriever, mock_client)
    
    # Assertions
    assert isinstance(docs, list)
    assert len(docs) > 0
    assert isinstance(metadata, dict)
    
    assert "mode" in metadata
    assert "dense_docs" in metadata
    assert "fused_docs" in metadata
    assert metadata["fused_docs"] == len(docs)
