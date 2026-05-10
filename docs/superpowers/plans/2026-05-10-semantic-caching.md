# Semantic Caching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement intent-based caching using query embeddings to save LLM calls while maintaining context awareness.

**Architecture:** 
- Extend `CacheManager` with a semantic registry (list of embeddings + metadata).
- Use `numpy` for fast linear scan similarity search.
- Integrate into `rag_pipeline` after the query rewrite step.

**Tech Stack:** Python, diskcache, numpy, FastEmbed.

---

### Task 1: Update `CacheManager` for Semantic Storage

**Files:**
- Modify: `backend/app/cache_manager.py`

- [ ] **Step 1: Implement `get_semantic` and `add_semantic` in `CacheManager`**

```python
# backend/app/cache_manager.py (Add to class CacheManager)

    def get_semantic(self, query_embedding: list[float], threshold: float = 0.95) -> Optional[str]:
        """Find a similar query in the semantic registry and return its cache key."""
        if self.cache_type != "diskcache" or self.disk_cache is None:
            return None
            
        registry = self.disk_cache.get("semantic_registry", [])
        if not registry:
            return None
            
        import numpy as np
        query_vec = np.array(query_embedding)
        
        best_score = -1.0
        best_key = None
        
        for entry in registry:
            entry_vec = np.array(entry["vector"])
            # Cosine similarity (assuming vectors are normalized, but safe to do full formula)
            norm_a = np.linalg.norm(query_vec)
            norm_b = np.linalg.norm(entry_vec)
            if norm_a == 0 or norm_b == 0:
                continue
            score = np.dot(query_vec, entry_vec) / (norm_a * norm_b)
            
            if score > best_score:
                best_score = score
                best_key = entry["key"]
                
        if best_score >= threshold:
            logger.info(f"Semantic Cache HIT: score={best_score:.4f}")
            return best_key
            
        return None

    def add_semantic(self, query_embedding: list[float], cache_key: str):
        """Add a new query embedding to the semantic registry."""
        if self.cache_type != "diskcache" or self.disk_cache is None:
            return
            
        registry = self.disk_cache.get("semantic_registry", [])
        registry.append({
            "vector": query_embedding,
            "key": cache_key
        })
        # Keep registry size manageable for linear scan
        if len(registry) > 1000:
            registry.pop(0)
        self.disk_cache.set("semantic_registry", registry)
```

- [ ] **Step 2: Commit Changes**

Run: `git add backend/app/cache_manager.py && git commit -m "feat: add semantic registry to CacheManager"`

---

### Task 2: Integrate Semantic Cache into RAG Pipeline

**Files:**
- Modify: `backend/app/rag_pipeline.py`

- [ ] **Step 1: Update `rag_pipeline` (sync version)**

```python
# backend/app/rag_pipeline.py
# ... inside rag_pipeline function ...

            standalone_query = rewrite_query(question, history)
            
            # --- NEW: Semantic Cache Check ---
            from .embeddings import get_embeddings
            embedder = get_embeddings()
            query_embedding = embedder.embed_query(standalone_query)
            
            semantic_key = cache.get_semantic(query_embedding)
            if semantic_key:
                cached_sem = cache.get(semantic_key)
                if cached_sem:
                    logger.info(f"RAG Semantic Cache HIT for: {standalone_query[:50]}")
                    # Log metrics similar to quick cache hit...
                    cache.set(quick_key, cached_sem, ttl=3600)
                    return cached_sem
            # --------------------------------
            
            standalone_normalized = standalone_query.strip().lower()
            # ... (proceed to deep cache check and generation) ...
            
            # After successful generation:
            cache.add_semantic(query_embedding, deep_key)
```

- [ ] **Step 2: Update `rag_pipeline_stream_async` (async version)**

```python
# backend/app/rag_pipeline.py
# ... inside rag_pipeline_stream_async ...

        standalone_query = await rewrite_query_async(query, chat_history)
        
        # --- NEW: Semantic Cache Check ---
        from .embeddings import get_embeddings
        embedder = get_embeddings()
        query_embedding = embedder.embed_query(standalone_query)
        
        semantic_key = cache.get_semantic(query_embedding)
        if semantic_key:
             cached_sem = cache.get(semantic_key)
             if cached_sem:
                 # yield step cached and done...
                 cache.set(quick_key, cached_sem, ttl=3600)
                 return
        # --------------------------------
```

- [ ] **Step 3: Commit Changes**

Run: `git add backend/app/rag_pipeline.py && git commit -m "feat: integrate semantic caching into RAG pipeline"`

---

### Task 3: Verification

- [ ] **Step 1: Test with similar queries**
Ask "Kapan Tim Cook lahir?" then "Berapa umur Tim Cook?" (assuming RAG returns birth date for both). Verify that the second query results in a "Semantic Cache HIT" in logs.

- [ ] **Step 2: Verify metrics**
Check the dashboard to see if response times for semantic hits are low (< 500ms).
