import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from app.index_data import process_pdf_documents

@patch("app.index_data.Path")
@patch("app.index_data.OpenDataLoaderPDFLoader")
@patch("app.index_data.MarkdownHeaderTextSplitter")
@patch("app.index_data.RecursiveCharacterTextSplitter")
def test_process_pdf_documents_uses_opendataloader_and_markdown_splitting(
    mock_recursive_splitter_class,
    mock_markdown_splitter_class,
    mock_loader_class,
    mock_path_class
):
    # Setup Mocks
    mock_path_instance = mock_path_class.return_value
    mock_path_instance.exists.return_value = True
    mock_path_instance.name = "financial_report.pdf"
    
    mock_loader = mock_loader_class.return_value
    mock_loader.load.return_value = [
        Document(page_content="# Financial Report\n## Section 1\nSome data", metadata={"page": 0})
    ]
    
    # Mock MarkdownHeaderTextSplitter behavior
    mock_markdown_splitter = mock_markdown_splitter_class.return_value
    mock_markdown_splitter.split_text.return_value = [
        Document(page_content="## Section 1\nSome data", metadata={"Header 1": "Financial Report"})
    ]
    
    # Mock RecursiveCharacterTextSplitter behavior
    mock_recursive_splitter = mock_recursive_splitter_class.from_language.return_value
    mock_recursive_splitter.create_documents.return_value = [
        Document(page_content="Some data", metadata={"Header 1": "Financial Report"})
    ]
    
    # Execute
    pdf_path = "financial_report.pdf"
    chunks = process_pdf_documents(pdf_path)
    
    # Assertions
    assert mock_loader_class.called, "OpenDataLoaderPDFLoader should be called"
    assert mock_markdown_splitter_class.called, "MarkdownHeaderTextSplitter should be called"
    
    # Verify metadata merging
    assert len(chunks) > 0
    first_chunk = chunks[0]
    assert first_chunk.metadata["source_file"] == "financial_report.pdf"
    assert first_chunk.metadata["page_number"] == 1
    assert first_chunk.metadata["Header 1"] == "Financial Report"
