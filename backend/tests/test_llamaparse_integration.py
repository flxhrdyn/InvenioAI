import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document as LangChainDocument
from app.index_data import process_pdf_documents

@patch("app.index_data.Path")
@patch("llama_parse.LlamaParse")
@patch("app.index_data.MarkdownHeaderTextSplitter")
@patch("app.index_data.RecursiveCharacterTextSplitter")
def test_process_pdf_documents_uses_llamaparse_and_markdown_splitting(
    mock_recursive_splitter_class,
    mock_markdown_splitter_class,
    mock_llamaparse_class,
    mock_path_class
):
    # Setup Mocks
    mock_path_instance = mock_path_class.return_value
    mock_path_instance.exists.return_value = True
    mock_path_instance.name = "financial_report.pdf"
    
    # Mock LlamaParse behavior
    mock_parser = mock_llamaparse_class.return_value
    
    # Mock Document object from LlamaIndex
    mock_llama_doc = MagicMock()
    mock_llama_doc.text = "# Financial Report\n## Section 1\nSome data"
    mock_llama_doc.metadata = {"page_number": "1"}
    
    mock_parser.load_data.return_value = [mock_llama_doc]
    
    # Mock MarkdownHeaderTextSplitter behavior
    mock_markdown_splitter = mock_markdown_splitter_class.return_value
    mock_markdown_splitter.split_text.return_value = [
        LangChainDocument(page_content="## Section 1\nSome data", metadata={"Header 1": "Financial Report"})
    ]
    
    # Mock RecursiveCharacterTextSplitter behavior
    mock_recursive_splitter_class.return_value.split_documents.return_value = [
        LangChainDocument(page_content="Some data", metadata={
            "Header 1": "Financial Report",
            "source": "financial_report.pdf",
            "source_file": "financial_report.pdf",
            "page_label": "1",
            "file_name": "financial_report.pdf"
        })
    ]
    
    # Execute
    pdf_path = "financial_report.pdf"
    chunks = process_pdf_documents(pdf_path)
    
    # Assertions
    assert mock_llamaparse_class.called, "LlamaParse should be called"
    assert mock_markdown_splitter_class.called, "MarkdownHeaderTextSplitter should be called"
    
    # Verify metadata
    assert len(chunks) > 0
    first_chunk = chunks[0]
    assert first_chunk.metadata["source_file"] == "financial_report.pdf"
    assert first_chunk.metadata["page_label"] == "1"
    assert first_chunk.metadata["Header 1"] == "Financial Report"
