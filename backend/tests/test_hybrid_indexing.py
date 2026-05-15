import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from backend.app.index_data import process_pdf_documents

@patch("backend.app.index_data.OpenDataLoaderPDFLoader")
def test_process_pdf_documents_uses_hybrid_mode(mock_loader_class):
    # Setup mock
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = []
    mock_loader_class.return_value = mock_loader_instance
    
    # Create a dummy file for the test
    dummy_file = "test_dummy.pdf"
    with open(dummy_file, "w") as f:
        f.write("dummy content")
    
    try:
        # Execute the function
        process_pdf_documents(dummy_file)
        
        # Verify OpenDataLoaderPDFLoader was called with hybrid="docling-fast"
        args, kwargs = mock_loader_class.call_args
        assert kwargs.get("hybrid") == "docling-fast", "OpenDataLoaderPDFLoader should be initialized with hybrid='docling-fast'"
        
    finally:
        # Cleanup
        if Path(dummy_file).exists():
            Path(dummy_file).unlink()

if __name__ == "__main__":
    pytest.main([__file__])
