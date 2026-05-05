"""Unit tests for small helper utilities.

Currently this module focuses on :func:`app.utils.format_docs`.
"""
from app.utils import format_docs


class MockDocument:
    """Minimal document object that matches what `format_docs` expects."""
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


def test_format_docs_single_document():
    """Format a single document and keep the source list in sync."""
    doc = MockDocument("This is test content", {"source": "test.pdf"})
    context, sources = format_docs([doc])
    
    assert context == "This is test content"
    assert sources == "- test.pdf"


def test_format_docs_multiple_documents():
    """Join multiple documents and emit one source line per doc."""
    docs = [
        MockDocument("First document", {"source": "doc1.pdf"}),
        MockDocument("Second document", {"source": "doc2.pdf"}),
        MockDocument("Third document", {"source": "doc3.pdf"})
    ]
    
    context, sources = format_docs(docs)
    
    # Check context contains all documents separated by double newline
    assert "First document" in context
    assert "Second document" in context
    assert "Third document" in context
    assert "\n\n" in context
    
    # Check sources
    assert "- doc1.pdf" in sources
    assert "- doc2.pdf" in sources
    assert "- doc3.pdf" in sources


def test_format_docs_empty_list():
    """Empty input returns empty strings."""
    context, sources = format_docs([])
    
    assert context == ""
    assert sources == ""


def test_format_docs_missing_source():
    """Missing metadata falls back to `unknown`."""
    doc = MockDocument("Content without source", {})
    context, sources = format_docs([doc])
    
    assert context == "Content without source"
    assert sources == "- unknown"


def test_format_docs_with_special_characters():
    """Newlines/tabs are preserved in the joined context."""
    doc = MockDocument("Content with\nnewlines\tand\ttabs", {"source": "file with spaces.pdf"})
    context, sources = format_docs([doc])
    
    assert "Content with\nnewlines\tand\ttabs" in context
    assert "- file with spaces.pdf" in sources


def test_format_docs_preserves_order():
    """Order is preserved in both context and source list."""
    docs = [
        MockDocument(f"Document {i}", {"source": f"doc{i}.pdf"})
        for i in range(5)
    ]
    
    context, sources = format_docs(docs)
    
    # Check order in context
    for i in range(5):
        assert f"Document {i}" in context
    
    # Check order in sources
    for i in range(5):
        assert f"- doc{i}.pdf" in sources


class TestFormatDocsEdgeCases:
    """Edge cases for `format_docs` input handling."""
    
    def test_very_long_content(self):
        """Very long content is returned unchanged."""
        long_content = "A" * 10000
        doc = MockDocument(long_content, {"source": "long.pdf"})
        context, sources = format_docs([doc])
        
        assert len(context) == 10000
        assert context == long_content
    
    def test_unicode_content(self):
        """Unicode text is handled without raising."""
        doc = MockDocument("Hello 世界 🌍", {"source": "unicode.pdf"})
        context, sources = format_docs([doc])
        
        assert "Hello 世界 🌍" in context
    
    def test_metadata_with_extra_fields(self):
        """Extra metadata keys are ignored by the formatter."""
        doc = MockDocument("Content", {
            "source": "test.pdf",
            "page": 5,
            "author": "Test Author",
            "extra_field": "value"
        })
        context, sources = format_docs([doc])
        
        assert context == "Content"
        assert sources == "- test.pdf"
