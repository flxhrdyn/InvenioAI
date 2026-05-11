"""Embedding model loader.

The embedding model is expensive to initialize, so we keep a single instance in
memory for the lifetime of the process.
"""

from __future__ import annotations

from functools import lru_cache

# Import config before importing the HF stack so environment defaults (timeouts,
# offline mode) are applied early.
from . import config as _config  # noqa: F401
from .config import EMBEDDING_MODEL, SPARSE_MODEL_NAME

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from fastembed import SparseTextEmbedding

@lru_cache(maxsize=1)
def get_embeddings() -> FastEmbedEmbeddings:
    """Return a cached dense embedding model instance."""
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)

class FastEmbedSparse:
    """Wrapper for FastEmbed SparseTextEmbedding to provide a consistent interface."""
    def __init__(self, model_name: str):
        self.model = SparseTextEmbedding(model_name=model_name)
    
    def embed_query(self, text: str):
        """Generate sparse embeddings for a single query."""
        results = self.model.embed([text])
        for r in results:
            yield {"indices": r.indices.tolist(), "values": r.values.tolist()}

    def embed_documents(self, texts: list[str]):
        """Generate sparse embeddings for multiple documents."""
        results = self.model.embed(texts)
        for r in results:
            yield {"indices": r.indices.tolist(), "values": r.values.tolist()}

@lru_cache(maxsize=1)
def get_sparse_embeddings() -> FastEmbedSparse:
    """Return a cached sparse embedding model instance."""
    return FastEmbedSparse(model_name=SPARSE_MODEL_NAME)