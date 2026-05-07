"""RAG pipeline orchestration.

The end-to-end flow is:

rewrite query → retrieve (dense or hybrid) → rerank → generate answer → return metrics.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
import hashlib
from .cache_manager import CacheManager
from typing import Any

from langchain_groq import ChatGroq

from .config import GROQ_API_KEY, LLM_MODEL, RETRIEVAL_K
from .qdrant_conn import close_qdrant_client, is_qdrant_client_closed_error
from .reranker import rerank
from .retriever import build_retriever, retrieve_documents, retrieve_documents_async
import json
from .utils import format_docs
from .metrics import log_query


logger = logging.getLogger(__name__)


QUERY_REWRITE_PROMPT = """
Rewrite the question into a standalone question.

Chat History:
{history}

Question:
{question}

Standalone Question:
"""


RAG_PROMPT = """
Answer the question using ONLY the context.

Context:
{context}

Question:
{question}

Answer in the same language.

Sources:
{sources}
"""


@lru_cache(maxsize=1)
def _get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY belum di-set. Isi di .env (lihat .env.example) sebelum menjalankan query."
        )

    return ChatGroq(
        model=LLM_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0,
    )

_cache_manager: CacheManager | None = None

def get_cache_manager() -> CacheManager:
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

def _get_cache_key(question: str, history: Any) -> str:
    """Generate a unique hash key for a query and its history."""
    hist_str = str(history) if history else ""
    raw = f"{question.strip().lower()}:{hist_str}"
    return f"rag_cache:{hashlib.md5(raw.encode()).hexdigest()}"


def format_history(history: Any) -> str:
    """Format chat history into a prompt-friendly string."""
    if not history:
        return ""

    if isinstance(history, str):
        return history

    return "\n".join(str(item) for item in history)


def rewrite_query(question: str, history: Any) -> str:
    """Rewrite a question into a standalone query."""
    # We can cache the rewrite too if we want, but it's usually fast enough (1s)
    prompt = QUERY_REWRITE_PROMPT.format(
        history=format_history(history),
        question=question,
    )
    return _get_llm().invoke(prompt).content




async def rewrite_query_async(query: str, history: list[str]) -> str:
    """Rewrite query to make it standalone using context asynchronously."""
    if not history:
        return query

    history_text = "\n".join([f"- {msg}" for msg in history[-3:]])
    prompt = QUERY_REWRITE_PROMPT.format(question=query, history=history_text)

    try:
        llm = _get_llm()
        response = await llm.ainvoke(prompt)
        content = response.content
        if isinstance(content, str):
            rewritten = content.strip()
            return rewritten if rewritten else query
        return query
    except Exception as e:
        logger.error(f"Query rewrite failed: {e}")
        return query

def rag_pipeline(question: str, history: Any) -> dict[str, Any]:
    """Run the RAG pipeline with two-layer caching.

    Returns a dict containing the answer, a sources string, and timing metrics.
    """
    total_start = time.monotonic()
    cache = get_cache_manager()
    quick_key = _get_cache_key(question, history)
    
    # Layer 1: Quick Cache (Exact request match)
    cached = cache.get(quick_key)
    if cached:
        logger.info(f"RAG Quick Cache Hit: {question[:50]}...")
        # Log to metrics so it appears in dashboard
        try:
            from .metrics import log_query
            log_query(
                question=question,
                response_time=0.01, # Minimal time for cache hit
                answer_length=len(cached.get("answer", "")),
                retrieval_time=0,
                generation_time=0,
                docs_retrieved=cached.get("metrics", {}).get("docs_retrieved", 0),
                chunks_processed=cached.get("metrics", {}).get("chunks_processed", 0),
                retrieval_scores=cached.get("metrics", {}).get("retrieval_scores", []),
            )
        except Exception:
            logger.warning("Failed to log cached query metrics", exc_info=True)
        return cached

    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            # Step 1: Rewrite Query
            standalone_query = rewrite_query(question, history)
            deep_key = f"rag_deep_cache:{hashlib.md5(standalone_query.strip().lower().encode()).hexdigest()}"
            
            # Layer 2: Deep Cache (Standalone query match)
            cached_deep = cache.get(deep_key)
            if cached_deep:
                logger.info(f"RAG Deep Cache Hit: {standalone_query[:50]}...")
                # Log to metrics
                try:
                    from .metrics import log_query
                    log_query(
                        question=question,
                        response_time=time.monotonic() - total_start,
                        answer_length=len(cached_deep.get("answer", "")),
                        retrieval_time=0,
                        generation_time=0,
                        docs_retrieved=cached_deep.get("metrics", {}).get("docs_retrieved", 0),
                        chunks_processed=cached_deep.get("metrics", {}).get("chunks_processed", 0),
                        retrieval_scores=cached_deep.get("metrics", {}).get("retrieval_scores", []),
                    )
                except Exception:
                    logger.warning("Failed to log deep cached query metrics", exc_info=True)
                # Save to Quick Cache for next time
                cache.set(quick_key, cached_deep, ttl=3600)
                return cached_deep

            # Full Pipeline
            result = _run_rag_pipeline_with_query(standalone_query, question, history)
            
            # Save to both caches
            cache.set(deep_key, result, ttl=3600)
            cache.set(quick_key, result, ttl=3600)
            return result
        except Exception as exc:
            if attempt < (max_attempts - 1) and is_qdrant_client_closed_error(exc):
                logger.warning("Qdrant client was closed during query; recreating and retrying once")
                close_qdrant_client()
                continue
            raise

    raise RuntimeError("Unexpected RAG retry state")

def _run_rag_pipeline_with_query(standalone_query: str, original_question: str, history: Any) -> dict[str, Any]:
    """Internal helper to run the core RAG steps once we have a standalone query."""
    total_start = time.monotonic()
    retriever, vectorstore, client = build_retriever()

    retrieval_start = time.monotonic()
    retrieved_docs, retrieval_meta = retrieve_documents(
        standalone_query,
        dense_retriever=retriever,
        client=client,
    )

    reranked_docs, retrieval_scores = rerank(standalone_query, retrieved_docs)
    retrieval_time = time.monotonic() - retrieval_start

    # Per-document similarity scores for the dashboard.
    if not retrieval_scores:
        try:
            scored = vectorstore.similarity_search_with_relevance_scores(
                standalone_query, k=RETRIEVAL_K
            )
            retrieval_scores = [round(float(score), 4) for _, score in scored]
        except Exception:
            logger.debug("Failed to fetch relevance scores", exc_info=True)
            retrieval_scores = []

    context, sources_str, sources_json = format_docs(reranked_docs)
    prompt = RAG_PROMPT.format(context=context, question=original_question, sources=sources_str)

    generation_start = time.monotonic()
    answer_msg = _get_llm().invoke(prompt)
    generation_time = time.monotonic() - generation_start

    total_time = time.monotonic() - total_start

    res = {
        "answer": answer_msg.content,
        "sources": sources_json,
        "metrics": {
            "total_time": round(total_time, 2),
            "retrieval_time": round(retrieval_time, 2),
            "generation_time": round(generation_time, 2),
            "docs_retrieved": len(retrieved_docs),
            "chunks_processed": len(reranked_docs),
            "retrieval_scores": retrieval_scores,
            "retrieval_mode": retrieval_meta.get("mode", "dense"),
            "dense_candidates": retrieval_meta.get("dense_docs", len(retrieved_docs)),
            "lexical_candidates": retrieval_meta.get("lexical_docs", 0),
            "fused_candidates": retrieval_meta.get("fused_docs", len(retrieved_docs)),
        },
    }

    # Log metrics to disk
    try:
        from .metrics import log_query
        log_query(
            question=original_question,
            response_time=total_time,
            answer_length=len(res["answer"]),
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            docs_retrieved=len(retrieved_docs),
            chunks_processed=len(reranked_docs),
            retrieval_scores=retrieval_scores,
        )
    except Exception:
        logger.warning("Failed to log query metrics", exc_info=True)

    return res

async def rag_pipeline_stream_async(query: str, chat_history: list[str]):
    from .retriever import build_retriever, retrieve_documents_async
    from .qdrant_conn import is_qdrant_client_closed_error, close_qdrant_client
    from .reranker import rerank
    from .config import RETRIEVAL_K
    import time

    total_start = time.monotonic()
    cache = get_cache_manager()
    quick_key = _get_cache_key(query, chat_history)
    logger.info(f"DEBUG: quick_key generation - query='{query}', history_len={len(chat_history)}")

    # Layer 1: Quick Cache
    cached = cache.get(quick_key)
    if cached:
        logger.info(f"RAG Stream Quick Cache Hit: {query[:50]}...")
        # Log to metrics
        try:
            from .metrics import log_query
            log_query(
                question=query,
                response_time=0.01,
                answer_length=len(cached.get("answer", "")),
                retrieval_time=0,
                generation_time=0,
                docs_retrieved=cached.get("metrics", {}).get("docs_retrieved", 0),
                chunks_processed=cached.get("metrics", {}).get("chunks_processed", 0),
                retrieval_scores=cached.get("metrics", {}).get("retrieval_scores", []),
            )
        except Exception:
            logger.warning("Failed to log cached stream metrics", exc_info=True)
            
        yield json.dumps({"step": "cached", "answer": cached["answer"]}) + "\n"
        yield json.dumps({
            "step": "done",
            "answer": cached["answer"],
            "sources": cached["sources"],
            "metrics": cached.get("metrics", {})
        }) + "\n"
        return

    try:
        # Step 1: Rewrite Query asynchronously
        yield json.dumps({"step": "rewriting"}) + "\n"
        standalone_query = await rewrite_query_async(query, chat_history)
        
        # Layer 2: Deep Cache
        standalone_hash = hashlib.md5(standalone_query.strip().lower().encode()).hexdigest()
        deep_key = f"rag_deep_cache:{standalone_hash}"
        logger.info(f"DEBUG: standalone_query='{standalone_query}' (hash={standalone_hash})")
        cached_deep = cache.get(deep_key)
        if cached_deep:
            logger.info(f"RAG Stream Deep Cache Hit: {standalone_query[:50]}...")
            answer = cached_deep.get("answer", "")
            sources = cached_deep.get("sources", [])
            m = cached_deep.get("metrics", {})
            
            yield json.dumps({
                "step": "done",
                "answer": answer,
                "sources": sources,
                "metrics": m
            }) + "\n"
            
            # Log metrics even for cache hits
            try:
                from .metrics import log_query
                log_query(
                    question=query,
                    response_time=time.monotonic() - total_start,
                    answer_length=len(answer),
                    retrieval_time=m.get("retrieval_time", 0),
                    generation_time=m.get("generation_time", 0),
                    docs_retrieved=len(sources) if isinstance(sources, list) else 0,
                    chunks_processed=m.get("reranked_docs", 0),
                    retrieval_scores=m.get("retrieval_scores", []),
                )
            except Exception:
                pass
            return

        # Step 2: Retrieval
        yield json.dumps({"step": "retrieving", "query": standalone_query}) + "\n"
        retriever, vectorstore, client = build_retriever()
        
        retrieval_start = time.monotonic()
        try:
            docs, metadata = await retrieve_documents_async(
                standalone_query,
                dense_retriever=retriever,
                client=client,
            )
        except Exception as e:
            if is_qdrant_client_closed_error(e):
                logger.info("Qdrant client closed, attempting to reopen...")
                close_qdrant_client()
                retriever, vectorstore, client = build_retriever()
                docs, metadata = await retrieve_documents_async(
                    standalone_query,
                    dense_retriever=retriever,
                    client=client,
                )
            else:
                raise
        retrieval_time = time.monotonic() - retrieval_start

        if not docs:
            # Log metrics even for failures
            try:
                from .metrics import log_query
                log_query(
                    question=query,
                    response_time=time.monotonic() - total_start,
                    answer_length=0,
                    retrieval_time=retrieval_time,
                    generation_time=0,
                    docs_retrieved=0,
                    chunks_processed=0,
                    retrieval_scores=[],
                )
            except Exception:
                pass

            yield json.dumps({
                "step": "done",
                "answer": "Maaf, saya tidak menemukan informasi relevan mengenai pertanyaan tersebut di dalam dokumen.",
                "docs": [],
                "metadata": metadata
            }) + "\n"
            return

        # Step 3: Reranking
        yield json.dumps({"step": "reranking"}) + "\n"
        top_docs, retrieval_scores = rerank(standalone_query, docs)
        metadata["retrieval_scores"] = retrieval_scores
        metadata["reranked_docs"] = len(top_docs)

        # Step 4: LLM Answer Generation (Streaming)
        yield json.dumps({"step": "generating"}) + "\n"
        
        # Format sources for the frontend
        from .utils import format_docs
        context_text, sources_str, sources_json = format_docs(top_docs)
        prompt = RAG_PROMPT.format(context=context_text, question=standalone_query, sources=sources_str)
        
        llm = _get_llm()
        generation_start = time.monotonic()
        full_answer = ""
        async for chunk in llm.astream(prompt):
            if chunk.content:
                full_answer += chunk.content
                yield json.dumps({"step": "token", "content": chunk.content}) + "\n"
        generation_time = time.monotonic() - generation_start
        
        total_time = time.monotonic() - total_start

        # Final result with sources
        final_result = {
            "step": "done",
            "answer": full_answer,
            "sources": sources_json,
            "metrics": {
                **metadata,
                "total_time": round(total_time, 2),
                "retrieval_time": round(retrieval_time, 2),
                "generation_time": round(generation_time, 2),
            }
        }
        
        # Cache the result for future identical queries (Both layers)
        cache_data = {
            "answer": full_answer,
            "sources": sources_json,
            "metrics": final_result["metrics"]
        }
        cache.set(deep_key, cache_data, ttl=3600)
        cache.set(quick_key, cache_data, ttl=3600)
        
        yield json.dumps(final_result) + "\n"

        # Log metrics to disk
        from .metrics import log_query
        log_query(
            question=query,
            response_time=total_time,
            answer_length=len(full_answer),
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            docs_retrieved=len(docs),
            chunks_processed=len(top_docs),
            retrieval_scores=metadata.get("retrieval_scores", []),
        )

    except Exception as e:
        logger.exception("Pipeline error")
        yield json.dumps({"step": "error", "message": str(e)}) + "\n"
