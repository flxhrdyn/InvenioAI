"""Sanity checks for configuration defaults.

These tests validate that key settings in :mod:`app.config` exist and that
their basic relationships make sense (e.g. overlap < chunk size).
"""
from app.config import (
    QDRANT_PATH,
    QDRANT_COLLECTION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_K,
    RERANK_TOP_K,
    USE_HYBRID_SEARCH,
    HYBRID_LEXICAL_K,
    HYBRID_FUSION_LIMIT,
    HYBRID_MAX_LEXICAL_DOCS,
    HYBRID_RRF_K,
    HYBRID_DENSE_WEIGHT,
    HYBRID_LEXICAL_WEIGHT,
    LLM_MODEL,
    EMBEDDING_MODEL
)


def test_qdrant_path_exists():
    """Qdrant path is defined (used for local/persistent mode)."""
    assert QDRANT_PATH is not None
    assert isinstance(QDRANT_PATH, str)
    assert len(QDRANT_PATH) > 0


def test_qdrant_collection_exists():
    """Collection name is defined."""
    assert QDRANT_COLLECTION is not None
    assert isinstance(QDRANT_COLLECTION, str)
    assert len(QDRANT_COLLECTION) > 0


def test_chunk_parameters():
    """Chunking settings are well-formed."""
    assert isinstance(CHUNK_SIZE, int)
    assert isinstance(CHUNK_OVERLAP, int)
    assert CHUNK_SIZE > 0
    assert CHUNK_OVERLAP >= 0
    assert CHUNK_OVERLAP < CHUNK_SIZE, "Overlap should be less than chunk size"


def test_retrieval_parameters():
    """Retrieval settings are well-formed."""
    assert isinstance(RETRIEVAL_K, int)
    assert isinstance(RERANK_TOP_K, int)
    assert RETRIEVAL_K > 0
    assert RERANK_TOP_K > 0
    assert RERANK_TOP_K <= RETRIEVAL_K, "Rerank top k should be <= retrieval k"


def test_hybrid_retrieval_parameters():
    """Hybrid retrieval settings are well-formed."""
    assert isinstance(USE_HYBRID_SEARCH, bool)
    assert isinstance(HYBRID_LEXICAL_K, int)
    assert isinstance(HYBRID_FUSION_LIMIT, int)
    assert isinstance(HYBRID_MAX_LEXICAL_DOCS, int)
    assert isinstance(HYBRID_RRF_K, int)
    assert isinstance(HYBRID_DENSE_WEIGHT, float)
    assert isinstance(HYBRID_LEXICAL_WEIGHT, float)

    assert HYBRID_LEXICAL_K > 0
    assert HYBRID_FUSION_LIMIT > 0
    assert HYBRID_MAX_LEXICAL_DOCS >= 100
    assert HYBRID_RRF_K > 0
    assert HYBRID_DENSE_WEIGHT >= 0
    assert HYBRID_LEXICAL_WEIGHT >= 0


def test_model_names():
    """Model identifiers are present."""
    assert LLM_MODEL is not None
    assert EMBEDDING_MODEL is not None
    assert isinstance(LLM_MODEL, str)
    assert isinstance(EMBEDDING_MODEL, str)
    assert len(LLM_MODEL) > 0
    assert len(EMBEDDING_MODEL) > 0


def test_api_key_handling():
    """API key can be missing in dev/test environments."""
    from app.config import GROQ_API_KEY
    # Should either be a string or None (if not set in .env)
    assert GROQ_API_KEY is None or isinstance(GROQ_API_KEY, str)


class TestConfigurationIntegrity:
    """Extra integrity checks for cross-field relationships."""
    
    def test_all_required_configs_exist(self):
        """Ensure the module exposes the expected set of config symbols."""
        required_configs = [
            'QDRANT_PATH',
            'QDRANT_COLLECTION',
            'CHUNK_SIZE',
            'CHUNK_OVERLAP',
            'RETRIEVAL_K',
            'RERANK_TOP_K',
            'LLM_MODEL',
            'EMBEDDING_MODEL'
        ]
        
        import app.config as config
        for conf in required_configs:
            assert hasattr(config, conf), f"Missing required config: {conf}"
    
    def test_numeric_configs_are_positive(self):
        """Numeric settings that represent sizes/counts are positive."""
        assert CHUNK_SIZE > 0
        assert RETRIEVAL_K > 0
        assert RERANK_TOP_K > 0
    
    def test_logical_relationships(self):
        """Validate simple logical relationships between related settings."""
        # Overlap should be smaller than chunk size
        assert CHUNK_OVERLAP < CHUNK_SIZE
        
        # Rerank top k should not exceed retrieval k
        assert RERANK_TOP_K <= RETRIEVAL_K
