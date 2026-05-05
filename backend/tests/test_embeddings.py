"""Tests for the embeddings factory.

Unit tests mock the underlying embedding class to avoid any model downloads.
Integration tests are opt-in via an environment variable.
"""
import os
from unittest.mock import MagicMock, patch

import pytest
from app.embeddings import get_embeddings


@patch("app.embeddings.FastEmbedEmbeddings")
def test_get_embeddings_returns_object(mock_fastembed):
    """Test that get_embeddings returns an embeddings object (no network)."""
    instance = MagicMock()
    mock_fastembed.return_value = instance

    embeddings = get_embeddings()
    assert embeddings is instance


@patch("app.embeddings.FastEmbedEmbeddings")
def test_embeddings_has_embed_query_method(mock_fastembed):
    """Test that embeddings object has embed_query method (no network)."""
    instance = MagicMock()
    instance.embed_query = MagicMock()
    mock_fastembed.return_value = instance

    embeddings = get_embeddings()
    assert hasattr(embeddings, "embed_query")
    assert callable(embeddings.embed_query)


@patch("app.embeddings.FastEmbedEmbeddings")
def test_embeddings_has_embed_documents_method(mock_fastembed):
    """Test that embeddings object has embed_documents method (no network)."""
    instance = MagicMock()
    instance.embed_documents = MagicMock()
    mock_fastembed.return_value = instance

    embeddings = get_embeddings()
    assert hasattr(embeddings, "embed_documents")
    assert callable(embeddings.embed_documents)


@patch("app.embeddings.FastEmbedEmbeddings")
def test_embed_query_returns_vector(mock_fastembed):
    """Test that embedding a query returns a vector (mocked)."""
    instance = MagicMock()
    instance.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])
    mock_fastembed.return_value = instance

    embeddings = get_embeddings()
    vector = embeddings.embed_query("Hello world")

    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(isinstance(x, float) for x in vector)


@patch("app.embeddings.FastEmbedEmbeddings")
def test_embed_documents_returns_vectors(mock_fastembed):
    """Test that embedding documents returns a list of vectors (mocked)."""
    instance = MagicMock()
    instance.embed_documents = MagicMock(return_value=[[0.1, 0.2], [0.2, 0.3]])
    mock_fastembed.return_value = instance

    embeddings = get_embeddings()
    docs = ["First document", "Second document"]
    vectors = embeddings.embed_documents(docs)

    assert isinstance(vectors, list)
    assert len(vectors) == 2
    assert all(isinstance(v, list) for v in vectors)


@patch("app.embeddings.FastEmbedEmbeddings")
def test_embed_query_consistent_dimensions(mock_fastembed):
    """Test that embeddings have consistent dimensions (mocked)."""
    instance = MagicMock()
    instance.embed_query = MagicMock(side_effect=[[0.1, 0.2], [0.2, 0.3]])
    mock_fastembed.return_value = instance

    embeddings = get_embeddings()
    vector1 = embeddings.embed_query("First text")
    vector2 = embeddings.embed_query("Second text")

    assert len(vector1) == len(vector2)


@patch("app.embeddings.FastEmbedEmbeddings")
def test_embed_empty_string(mock_fastembed):
    """Test embedding an empty string (mocked)."""
    instance = MagicMock()
    instance.embed_query = MagicMock(return_value=[0.0])
    mock_fastembed.return_value = instance

    embeddings = get_embeddings()
    vector = embeddings.embed_query("")

    assert isinstance(vector, list)
    assert len(vector) > 0


@patch("app.embeddings.FastEmbedEmbeddings")
def test_embed_special_characters(mock_fastembed):
    """Test embedding text with special characters (mocked)."""
    instance = MagicMock()
    instance.embed_query = MagicMock(return_value=[0.1])
    mock_fastembed.return_value = instance


    embeddings = get_embeddings()
    text = "Hello @world! #python $100 & more..."
    vector = embeddings.embed_query(text)

    assert isinstance(vector, list)
    assert len(vector) > 0


@pytest.mark.skipif(
    os.getenv("RUN_EMBEDDINGS_INTEGRATION") != "1",
    reason="Set RUN_EMBEDDINGS_INTEGRATION=1 to run tests that require downloading HF models.",
)
class TestEmbeddingsIntegration:
    """Opt-in integration tests that download a model."""
    
    def test_similar_texts_have_similar_embeddings(self):
        """Similar sentences should embed closer than unrelated text."""
        embeddings = get_embeddings()
        
        v1 = embeddings.embed_query("The cat sat on the mat")
        v2 = embeddings.embed_query("A cat is sitting on a mat")
        v3 = embeddings.embed_query("Python programming language")
        
        # Calculate simple cosine similarity
        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot_product / (norm_a * norm_b)
        
        sim_cat = cosine_similarity(v1, v2)
        sim_python = cosine_similarity(v1, v3)
        
        # Similar sentences should have higher similarity
        assert sim_cat > sim_python
    
    def test_multiple_get_embeddings_calls_work(self):
        """Multiple calls to `get_embeddings` remain usable."""
        emb1 = get_embeddings()
        emb2 = get_embeddings()
        
        v1 = emb1.embed_query("test")
        v2 = emb2.embed_query("test")
        
        # Both should work and produce same dimension
        assert len(v1) == len(v2)
