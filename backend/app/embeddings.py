"""Embedding model loader.

The embedding model is expensive to initialize, so we keep a single instance in
memory for the lifetime of the process.
"""

from __future__ import annotations

from functools import lru_cache

# Import config before importing the HF stack so environment defaults (timeouts,
# offline mode) are applied early.
from . import config as _config  # noqa: F401
from .config import EMBEDDING_MODEL

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

@lru_cache(maxsize=1)
def get_embeddings() -> FastEmbedEmbeddings:
    """Return a cached embedding model instance."""
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)