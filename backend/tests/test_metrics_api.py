import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.asyncio
async def test_get_metrics_endpoint():
    """Test that the /metrics endpoint returns a valid response."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/metrics")
    
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data
    assert "query_history" in data
    assert "avg_response_time" in data
