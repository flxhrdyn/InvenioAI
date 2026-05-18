import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document as LangChainDocument
from app.index_data import process_pdf_documents

@patch("app.index_data.Path")
@patch("llama_parse.LlamaParse")
@patch("app.index_data.MarkdownHeaderTextSplitter")
@patch("app.index_data.RecursiveCharacterTextSplitter")
def test_process_pdf_documents_extracts_text_and_metadata(
    mock_recursive_splitter_class,
    mock_markdown_splitter_class,
    mock_llamaparse_class,
    mock_path_class
):
    # Setup Mocks
    mock_path_instance = mock_path_class.return_value
    mock_path_instance.exists.return_value = True
    mock_path_instance.name = "dummy.pdf"
    
    mock_parser = mock_llamaparse_class.return_value
    mock_llama_doc = MagicMock()
    mock_llama_doc.text = "test PDF"
    mock_llama_doc.metadata = {"page_number": "1"}
    mock_parser.load_data.return_value = [mock_llama_doc]
    
    mock_markdown_splitter = mock_markdown_splitter_class.return_value
    mock_markdown_splitter.split_text.return_value = [
        LangChainDocument(page_content="test PDF", metadata={})
    ]
    
    mock_recursive_splitter = mock_recursive_splitter_class.return_value
    mock_recursive_splitter.split_documents.return_value = [
        LangChainDocument(page_content="test PDF", metadata={
            "source_file": "dummy.pdf",
            "page_label": "1"
        })
    ]
    
    # Execute
    pdf_path = "dummy.pdf"
    docs = process_pdf_documents(pdf_path)
    
    # Assertions
    assert len(docs) > 0
    first_doc = docs[0]
    assert "test PDF" in first_doc.page_content
    assert first_doc.metadata["source_file"] == "dummy.pdf"
    assert first_doc.metadata["page_label"] == "1"


@patch("app.index_data.Path")
@patch("llama_parse.LlamaParse")
@patch("app.index_data.MarkdownHeaderTextSplitter")
@patch("app.index_data.RecursiveCharacterTextSplitter")
def test_recursive_splitting_preserves_structure(
    mock_recursive_splitter_class,
    mock_markdown_splitter_class,
    mock_llamaparse_class,
    mock_path_class
):
    # Setup Mocks
    mock_path_instance = mock_path_class.return_value
    mock_path_instance.exists.return_value = True
    mock_path_instance.name = "dummy.pdf"
    
    mock_parser = mock_llamaparse_class.return_value
    mock_llama_doc = MagicMock()
    mock_llama_doc.text = "sentence one. sentence two."
    mock_llama_doc.metadata = {"page_number": "1"}
    mock_parser.load_data.return_value = [mock_llama_doc]
    
    mock_markdown_splitter = mock_markdown_splitter_class.return_value
    mock_markdown_splitter.split_text.return_value = [
        LangChainDocument(page_content="sentence one. sentence two.", metadata={})
    ]
    
    mock_recursive_splitter = mock_recursive_splitter_class.return_value
    mock_recursive_splitter.split_documents.return_value = [
        LangChainDocument(page_content="sentence one.", metadata={"source_file": "dummy.pdf", "page_label": "1"}),
        LangChainDocument(page_content="sentence two.", metadata={"source_file": "dummy.pdf", "page_label": "1"})
    ]
    
    # Execute
    pdf_path = "dummy.pdf"
    docs = process_pdf_documents(pdf_path)
    
    # Assertions
    assert len(docs) == 2
    assert docs[0].metadata["page_label"] == "1"
    assert docs[1].metadata["page_label"] == "1"
    for doc in docs:
        assert doc.metadata["source_file"] == "dummy.pdf"
