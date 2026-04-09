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

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover - compatibility fallback
    from langchain_community.embeddings import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached embedding model instance."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)