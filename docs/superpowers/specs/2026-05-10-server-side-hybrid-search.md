# Design Spec: Server-Side Hybrid Search with BM42

Implement a native hybrid search strategy in Qdrant using FastEmbed for both dense and sparse vectors. This eliminates the need for client-side BM25 indexing and improves scalability and performance.

## User Review Required

> [!IMPORTANT]
> **Data Loss Warning**: Implementing this change requires updating the Qdrant collection schema. Existing indexed data must be deleted and re-indexed to generate the new sparse vectors.

> [!NOTE]
> **Model Change**: We are switching from purely statistical BM25 (local) to BM42 (Attention-based sparse vectors). This remains language-agnostic and supports Indonesian well.

## Proposed Changes

### [Backend] Ingestion & Embeddings

#### [MODIFY] [embeddings.py](file:///d:/Programming/Python/Python%20Projects/01_GenAI/rag-invenio-ai/backend/app/embeddings.py)
- Introduce `SparseTextEmbedding` from FastEmbed.
- Create a singleton wrapper that provides both `dense_model` and `sparse_model`.
- Ensure models are cached in memory for performance.

#### [MODIFY] [index_api.py](file:///d:/Programming/Python/Python%20Projects/01_GenAI/rag-invenio-ai/backend/app/index_api.py) (or related indexing logic)
- Update the ingestion loop to generate both dense and sparse embeddings for each chunk.
- Update `client.upsert` to push named vectors: `{"dense": [...], "sparse": {"indices": [...], "values": [...]}}`.

### [Backend] Retrieval Strategy

#### [MODIFY] [retriever.py](file:///d:/Programming/Python/Python%20Projects/01_GenAI/rag-invenio-ai/backend/app/retriever.py)
- **Delete**: Remove `BM25Retriever` (local) and `_load_documents_for_bm25`.
- **Delete**: Remove `reciprocal_rank_fusion` (local client-side logic).
- **Update**: Configure `QdrantVectorStore` to use `RetrievalMode.HYBRID`.
- **Implement**: Use Qdrant's internal RRF or Weighted fusion.
- **Refinement**: Maintain FlashRank reranking on the final candidates returned by Qdrant.

### [Backend] Configuration

#### [MODIFY] [config.py](file:///d:/Programming/Python/Python%20Projects/01_GenAI/rag-invenio-ai/backend/app/config.py)
- Add `SPARSE_MODEL_NAME` (Default: `Qdrant/bm42-all-minilm-l6-v2-attentions`).
- Add weights for Hybrid search (e.g., `HYBRID_DENSE_WEIGHT` vs `HYBRID_SPARSE_WEIGHT`).

## Verification Plan

### Automated Tests
- Create a new test `test_hybrid_search_native.py` to verify Qdrant returns both semantic and keyword-based results.
- Verify that no local BM25 index is created/loaded in memory during retrieval.

### Manual Verification
- Re-index a sample Indonesian PDF.
- Search for specific technical terms/IDs to confirm BM42 keyword matching is working.
- Verify through logs that "Lexical docs" are now coming from Qdrant directly.
