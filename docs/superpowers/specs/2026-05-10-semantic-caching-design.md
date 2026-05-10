# Design Spec: Semantic Caching for RAG Pipeline

**Date**: 2026-05-10
**Status**: Draft
**Topic**: Improving Cache Hit Rate using Semantic Similarity

## Goal
Improve the cache hit rate of the InvenioAI RAG pipeline by implementing a semantic lookup layer. This ensures that paraphrased questions (with the same intent and context) reuse previous LLM generations, saving API costs and reducing response latency.

## Architecture

The system will use a hybrid approach:
1.  **Exact Cache (Quick)**: MD5 hash of `(question + history)` for instant hits on identical inputs.
2.  **Semantic Cache (Deep)**: Embedding-based similarity search of the `standalone_query`.

### Components

#### 1. Semantic Registry
A persistent list stored in `DiskCache` containing metadata for previous successful RAG runs:
- `embedding`: Vector representation of the standalone query.
- `query_text`: The original standalone query (for debugging).
- `cache_key`: The MD5 key used to retrieve the full result object.

#### 2. Lookup Logic
- **Input**: `standalone_query`
- **Process**:
    1. Generate embedding using `FastEmbed` (existing model).
    2. Load registry from `DiskCache`.
    3. Calculate cosine similarity using `numpy` against all stored vectors.
    4. Find the entry with the highest score.
    5. **Threshold**: 0.95 (Initial).
- **Output**: Cached RAG result if score > Threshold, else `None`.

#### 3. Update Logic
- When a RAG pipeline finishes a fresh generation:
    1. Generate embedding for the `standalone_query`.
    2. Append `{embedding, query_text, cache_key}` to the registry.
    3. Save registry back to `DiskCache`.

## Context Awareness
The "Context Aware" requirement is satisfied by the **Query Rewriting** step. Since the semantic cache is keyed by the *standalone* (rewritten) query, it inherently includes the history context.
- Turn 1: "Who is the CEO of Apple?" -> Standalone: "Who is the CEO of Apple?"
- Turn 2: "When was **he** born?" -> Standalone: "When was **Tim Cook** born?"
- Parallel Turn 2: "When was **he** born?" (after discussing Satya Nadella) -> Standalone: "When was **Satya Nadella** born?"

Because the standalone queries are semantically distinct, the cache will correctly distinguish between them.

## Performance Considerations
- **Linear Scan**: For a portfolio app, the cache size is likely < 1000 entries. A linear scan with `numpy` for 1000 vectors takes ~1-2ms, which is negligible compared to an LLM call.
- **Memory**: `numpy` arrays are efficient. 1000 vectors (384 dimensions) = ~1.5 MB.

## Implementation Plan (Preview)
1.  Modify `cache_manager.py` to support semantic retrieval/storage.
2.  Update `rag_pipeline.py` to integrate the semantic lookup after the rewrite step.
3.  Add unit tests for semantic hit/miss scenarios.

## User Review Required

> [!IMPORTANT]
> The threshold of **0.95** is a heuristic. We might need to tune this if we get "false hits" (answering a slightly different question with a cached result).

> [!NOTE]
> Since this is a portfolio app, we will prioritize simplicity and use `numpy` for the vector math instead of adding a complex vector DB layer for the cache itself.
