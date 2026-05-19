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


def test_strip_running_headers_footers_removes_repetitive_boundaries():
    """Verify that repetitive top/bottom boundary lines across pages are stripped."""
    from app.index_data import strip_running_headers_footers
    class MockDoc:
        def __init__(self, text):
            self.text = text

    # Buat 5 mock pages dengan running header dan footer yang sama di top/bottom,
    # namun dengan isi body yang berbeda
    pages = [
        MockDoc("Running Header Title\n\nBody content of page 1.\nImportant Info 1\n\nPage Footer 2023"),
        MockDoc("Running Header Title\n\nBody content of page 2.\nImportant Info 2\n\nPage Footer 2023"),
        MockDoc("Running Header Title\n\nBody content of page 3.\nImportant Info 3\n\nPage Footer 2023"),
        MockDoc("Running Header Title\n\nBody content of page 4.\nImportant Info 4\n\nPage Footer 2023"),
        MockDoc("Running Header Title\n\nBody content of page 5.\nImportant Info 5\n\nPage Footer 2023"),
    ]

    cleaned = strip_running_headers_footers(pages)
    
    # Assert running headers and footers are fully removed
    for page in cleaned:
        assert "Running Header Title" not in page.text
        assert "Page Footer 2023" not in page.text
        # Assert body content is untouched
        assert "Body content of page" in page.text
        assert "Important Info" in page.text


def test_strip_running_headers_footers_ignores_markdown_headers_and_unique_content():
    """Verify that actual structural markdown headings and unique body content are never removed."""
    from app.index_data import strip_running_headers_footers
    class MockDoc:
        def __init__(self, text):
            self.text = text

    # Halaman yang menduplikasi heading asli atau isi body yang kebetulan berulang
    pages = [
        MockDoc("# 7. Income taxes\n\nUnique text on page 1.\n\nFooter"),
        MockDoc("# 7. Income taxes\n\nUnique text on page 2.\n\nFooter"),
        MockDoc("# 7. Income taxes\n\nUnique text on page 3.\n\nFooter"),
    ]

    cleaned = strip_running_headers_footers(pages)
    
    # Assert structural headings starting with # are NOT removed
    for page in cleaned:
        assert "# 7. Income taxes" in page.text


@patch("app.index_data.Path")
@patch("llama_parse.LlamaParse")
@patch("app.index_data.MarkdownHeaderTextSplitter")
@patch("app.index_data.RecursiveCharacterTextSplitter")
def test_header_inheritance_propagates_headers_across_pages(
    mock_recursive_splitter_class,
    mock_markdown_splitter_class,
    mock_llamaparse_class,
    mock_path_class
):
    """Verify that headers are propagated to consecutive pages without headings."""
    # Setup Mocks
    mock_path_instance = mock_path_class.return_value
    mock_path_instance.exists.return_value = True
    mock_path_instance.name = "dummy.pdf"
    
    mock_parser = mock_llamaparse_class.return_value
    
    # Page 1 has Header 1
    # Page 2 has no new header
    # Page 3 has a new Header 1
    mock_llama_doc_1 = MagicMock()
    mock_llama_doc_1.text = "# 7. Income taxes\n\nPage 1 body"
    mock_llama_doc_1.metadata = {"page_number": "1"}
    
    mock_llama_doc_2 = MagicMock()
    mock_llama_doc_2.text = "Page 2 body (continuation)"
    mock_llama_doc_2.metadata = {"page_number": "2"}
    
    mock_llama_doc_3 = MagicMock()
    mock_llama_doc_3.text = "# 8. Financial instruments\n\nPage 3 body"
    mock_llama_doc_3.metadata = {"page_number": "3"}
    
    mock_parser.load_data.return_value = [mock_llama_doc_1, mock_llama_doc_2, mock_llama_doc_3]
    
    # Configure Markdown splitter mock
    mock_markdown_splitter = mock_markdown_splitter_class.return_value
    
    # Page 1 split: has Header 1
    doc_1_split = LangChainDocument(page_content="Page 1 body", metadata={"Header 1": "7. Income taxes"})
    # Page 2 split: has empty metadata (no headers on page 2)
    doc_2_split = LangChainDocument(page_content="Page 2 body (continuation)", metadata={})
    # Page 3 split: has new Header 1
    doc_3_split = LangChainDocument(page_content="Page 3 body", metadata={"Header 1": "8. Financial instruments"})
    
    # split_text is called per page
    mock_markdown_splitter.split_text.side_effect = [
        [doc_1_split],
        [doc_2_split],
        [doc_3_split]
    ]
    
    # Configure Recursive splitter mock to just return whatever it is given
    mock_recursive_splitter = mock_recursive_splitter_class.return_value
    def mock_split_documents(docs):
        return docs
    mock_recursive_splitter.split_documents.side_effect = mock_split_documents
    
    # Execute
    docs = process_pdf_documents("dummy.pdf")
    
    # Assertions
    assert len(docs) == 3
    # Page 1 split should have Header 1
    assert docs[0].metadata["Header 1"] == "7. Income taxes"
    assert docs[0].metadata["page_label"] == "1"
    
    # Page 2 split should have inherited Header 1 from page 1
    assert docs[1].metadata["Header 1"] == "7. Income taxes"
    assert docs[1].metadata["page_label"] == "2"
    
    # Page 3 split should have the new Header 1 and not inherit the old one
    assert docs[2].metadata["Header 1"] == "8. Financial instruments"
    assert docs[2].metadata["page_label"] == "3"

