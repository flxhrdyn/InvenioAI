import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
@patch("app.rag_pipeline.rag_pipeline_stream_async")
async def test_query_stream_endpoint(mock_rag_pipeline_stream_async):
    # Setup mock to yield chunks
    async def mock_stream(*args, **kwargs):
        yield "chunk1"
        yield "chunk2"
    
    mock_rag_pipeline_stream_async.return_value = mock_stream()
    
    # Execute
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/query/stream", json={"question": "test query", "history": []})
    
    # Assertions
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    content = response.content.decode()
    assert "data: chunk1" in content
    assert "data: chunk2" in content
