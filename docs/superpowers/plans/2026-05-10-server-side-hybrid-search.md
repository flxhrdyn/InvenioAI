# Server-Side Hybrid Search (BM42) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace client-side BM25 indexing with Qdrant's native hybrid search using BM42 sparse vectors to improve scalability and retrieval quality.

**Architecture:** Transition from client-side `rank-bm25` to server-side Sparse Vectors. Use FastEmbed to generate both Dense and Sparse embeddings during ingestion and search. Combine results in Qdrant using RRF/Weighted fusion.

**Tech Stack:** FastEmbed (BM42), Qdrant (Sparse Vectors), LangChain-Qdrant.

---

### Task 1: Environment & Configuration

**Files:**
- Modify: `backend/app/config.py`

- [ ] **Step 1: Add sparse model and hybrid weight configurations**

Modify `backend/app/config.py` to include:
```python
# Hybrid Search Configuration
USE_HYBRID_SEARCH = True
SPARSE_MODEL_NAME = "Qdrant/bm42-all-minilm-l6-v2-attentions"
HYBRID_DENSE_WEIGHT = 0.5
HYBRID_SPARSE_WEIGHT = 0.5
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/config.py
git commit -m "feat: add hybrid search configuration constants"
```

---

### Task 2: Multi-Model Embedding Support

**Files:**
- Modify: `backend/app/embeddings.py`

- [ ] **Step 1: Write test for sparse embedding generation**

Create `backend/tests/test_embeddings_sparse.py`:
```python
from app.embeddings import get_sparse_embeddings

def test_sparse_embedding_output():
    model = get_sparse_embeddings()
    query = "halo dunia"
    vector = list(model.embed_query(query))
    assert len(vector) > 0
    assert isinstance(vector[0], dict) # FastEmbed sparse output format
    assert "indices" in vector[0]
    assert "values" in vector[0]
```

- [ ] **Step 2: Run test to verify failure**
Run: `pytest backend/tests/test_embeddings_sparse.py`
Expected: FAIL (ImportError or function not defined)

- [ ] **Step 3: Implement sparse embedding singleton**

Modify `backend/app/embeddings.py`:
```python
from fastembed import SparseTextEmbedding

_sparse_embeddings = None

def get_sparse_embeddings():
    global _sparse_embeddings
    if _sparse_embeddings is None:
        _sparse_embeddings = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    return _sparse_embeddings
```

- [ ] **Step 4: Run test to verify pass**
Run: `pytest backend/tests/test_embeddings_sparse.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add backend/app/embeddings.py backend/tests/test_embeddings_sparse.py
git commit -m "feat: implement sparse embedding singleton with FastEmbed"
```

---

### Task 3: Qdrant Collection with Sparse Support

**Files:**
- Modify: `backend/scripts/init_vector_db.py` (or ingestion logic)

- [ ] **Step 1: Update collection creation to include sparse_vectors_config**

Modify the collection creation logic:
```python
from qdrant_client.http import models

client.recreate_collection(
    collection_name=QDRANT_COLLECTION,
    vectors_config={
        "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams()
    }
)
```

- [ ] **Step 2: Commit**
```bash
git commit -m "feat: update qdrant collection schema for hybrid search"
```

---

### Task 4: Hybrid Ingestion Logic

**Files:**
- Modify: `backend/app/index_api.py`

- [ ] **Step 1: Update indexing to push both dense and sparse vectors**

Modify the `upsert` call in the indexing loop:
```python
# Pseudocode for index_api.py
dense_vec = dense_model.embed_documents([text])[0]
sparse_vec = list(sparse_model.embed_documents([text]))[0]

client.upsert(
    collection_name=QDRANT_COLLECTION,
    points=[
        models.PointStruct(
            id=doc_id,
            vector={
                "dense": dense_vec,
                "sparse": sparse_vec
            },
            payload=payload
        )
    ]
)
```

- [ ] **Step 2: Commit**
```bash
git commit -m "feat: implement dual-vector ingestion (dense + sparse)"
```

---

### Task 5: Native Hybrid Retriever

**Files:**
- Modify: `backend/app/retriever.py`

- [ ] **Step 1: Write integration test for hybrid retrieval**

Create `backend/tests/test_hybrid_retrieval.py`:
```python
from app.retriever import retrieve_documents
from app.qdrant_conn import get_qdrant_client

def test_hybrid_retrieval_returns_results():
    client = get_qdrant_client()
    # Assume some data is already indexed
    docs, meta = retrieve_documents("test query", dense_retriever=..., client=client)
    assert len(docs) > 0
    assert meta["mode"] == "hybrid-native"
```

- [ ] **Step 2: Remove legacy BM25 code and implement Native Hybrid**

Modify `backend/app/retriever.py`:
- Remove `BM25Retriever` imports and `_load_documents_for_bm25`.
- Update `retrieve_documents` to use `client.search` with `prefetch` for hybrid fusion.

```python
# New retrieval logic using Qdrant Prefetch for RRF
from qdrant_client.http import models

def retrieve_documents(query, ...):
    dense_vector = dense_model.embed_query(query)
    sparse_vector = list(sparse_model.embed_query(query))[0]
    
    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_filter=None,
        search_params=models.SearchParams(hnsw_ef=128, exact=False),
        prefetch=[
            models.Prefetch(vector="dense", query=dense_vector, limit=20),
            models.Prefetch(vector="sparse", query=sparse_vector, limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=RETRIEVAL_K
    )
    # Convert to Document objects
```

- [ ] **Step 3: Run integration test**
Run: `pytest backend/tests/test_hybrid_retrieval.py`
Expected: PASS

- [ ] **Step 4: Commit**
```bash
git commit -m "feat: replace client-side BM25 with native Qdrant hybrid search"
```

---

### Task 6: Cleanup & Final Verification

- [ ] **Step 1: Remove rank-bm25 dependency**
Modify `pyproject.toml` and `requirements.txt`.

- [ ] **Step 2: Final project-wide test run**
Run: `pytest backend/tests`

- [ ] **Step 3: Commit**
```bash
git commit -m "chore: cleanup legacy bm25 dependencies and verify all tests"
```
