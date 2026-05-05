import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from app.main import app

client = TestClient(app)

@patch("app.index_api.get_qdrant_client")
def test_delete_document_success(mock_get_qdrant):
    # Setup mock
    mock_qdrant = MagicMock()
    mock_get_qdrant.return_value = mock_qdrant
    
    # Mocking scroll to simulate the document exists (used in list_documents)
    # We'll just assume it exists for the DELETE call.
    # The actual implementation of DELETE /documents/{filename} will be added.
    
    filename = "test.pdf"
    
    # Execute
    response = client.delete(f"/documents/{filename}")
    
    # Assert
    assert response.status_code == 200
    assert response.json() == {"status": f"Document '{filename}' deleted successfully"}
    
    # Verify Qdrant was called with correct filter
    # Expecting delete(collection_name=..., points_selector=...)
    mock_qdrant.delete.assert_called_once()
    args, kwargs = mock_qdrant.delete.call_args
    assert kwargs["collection_name"] == "invenioai_collection"
    
    # Verify filter
    points_selector = kwargs["points_selector"]
    from qdrant_client.http import models
    assert isinstance(points_selector, models.FilterSelector)
    
    # We expect a filter that matches metadata.source_file
    filter_obj = points_selector.filter
    assert filter_obj.must[0].key == "metadata.source_file"
    assert filter_obj.must[0].match.value == filename
