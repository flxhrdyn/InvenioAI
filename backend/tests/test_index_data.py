import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from app.index_data import process_pdf_documents

@patch("app.index_data.Path")
@patch("app.index_data.PyMuPDFLoader")
@patch("app.index_data.RecursiveCharacterTextSplitter")
def test_process_pdf_documents_extracts_text_and_metadata(
    mock_splitter_class, mock_loader_class, mock_path_class
):
    # Setup Mocks
    mock_path_instance = mock_path_class.return_value
    mock_path_instance.exists.return_value = True
    mock_path_instance.name = "dummy.pdf"
    
    mock_loader = mock_loader_class.return_value
    mock_loader.load.return_value = [
        Document(page_content="test PDF", metadata={"page": 0})
    ]
    
    mock_splitter = mock_splitter_class.return_value
    mock_splitter.split_documents.return_value = [
        Document(page_content="test PDF", metadata={"page": 0})
    ]
    
    # Execute
    pdf_path = "dummy.pdf"
    docs = process_pdf_documents(pdf_path)
    
    # Assertions
    assert len(docs) > 0
    first_doc = docs[0]
    assert "test PDF" in first_doc.page_content
    assert "source_file" in first_doc.metadata
    assert first_doc.metadata["source_file"] == "dummy.pdf"
    # Verify page_number injection (0-indexed page becomes 1-indexed page_number)
    assert first_doc.metadata["page_number"] == 1


@patch("app.index_data.Path")
@patch("app.index_data.PyMuPDFLoader")
@patch("app.index_data.RecursiveCharacterTextSplitter")
def test_recursive_splitting_preserves_structure(
    mock_splitter_class, mock_loader_class, mock_path_class
):
    # Setup Mocks
    mock_path_instance = mock_path_class.return_value
    mock_path_instance.exists.return_value = True
    mock_path_instance.name = "dummy.pdf"
    
    mock_loader = mock_loader_class.return_value
    mock_loader.load.return_value = [
        Document(page_content="sentence one. sentence two.", metadata={"page": 0})
    ]
    
    mock_splitter = mock_splitter_class.return_value
    mock_splitter.split_documents.return_value = [
        Document(page_content="sentence one.", metadata={"page": 0}),
        Document(page_content="sentence two.", metadata={"page": 0})
    ]
    
    # Execute
    pdf_path = "dummy.pdf"
    docs = process_pdf_documents(pdf_path)
    
    # Assertions
    assert len(docs) == 2
    assert docs[0].metadata["page_number"] == 1
    assert docs[1].metadata["page_number"] == 1
    for doc in docs:
        assert "source_file" in doc.metadata
        assert doc.metadata["source_file"] == "dummy.pdf"
