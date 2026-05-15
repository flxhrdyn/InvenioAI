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
from .utils import format_docs, ThinkingParser
from .metrics import log_query
from .embeddings import get_embeddings


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
Answer the question using the provided context. Follow the instructions strictly.

DATA RULES:
1. **Table is King**: If a value is in a table, use it. ALWAYS double check the column header (e.g., 2023 vs 2022).
2. **Column Alignment**: If a table looks truncated, look for the row label and count columns carefully. 
3. **No Math**: Do NOT calculate if the final number is already there.
4. **Units**: $m = juta, $b = miliar (in Indonesian).

EXAMPLE OF CORRECT REASONING:
Context: "Table: | Item | 2023 | 2022 | \n | Revenue | 15,433 | 16,136 |"
User: "Berapa revenue 2023?"
<thinking>
Step 1: Entity is Revenue, Year is 2023.
Step 2: Table has two years: 2023 and 2022.
Step 3: Column 2023 corresponds to 15,433. Column 2022 corresponds to 16,136.
Step 4: The answer for 2023 is 15,433.
</thinking>
Revenue pada tahun 2023 adalah 15.433 juta.

Context:
{context}

Question:
{question}

Instructions:
1. START with <thinking>...</thinking>.
2. Inside, use 4 Steps: 1. Deconstruction, 2. Retrieval, 3. Synthesis (Check Table vs Text), 4. Strategy.
3. CLOSE the tag </thinking> before answering.
4. Provide a professional, narrative answer in the SAME LANGUAGE as the question.
5. DO NOT cite sources manually at the end (e.g., no "Sumber: ..." or "Source: ..."). The UI will handle this.

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


def format_history(history: Any, max_items: int = 5) -> str:
    """Format chat history into a prompt-friendly string, limited to last N items."""
    if not history:
        return ""

    if isinstance(history, str):
        return history

    # If it's a list, take the last max_items
    items = list(history)[-max_items:]
    return "\n".join(str(item) for item in items)


def rewrite_query(question: str, history: Any) -> str:
    """Rewrite a question into a standalone query with local caching."""
    # Check if we have a cached standalone version for this question + simplified history
    # For standalone questions (empty history), we can cache the result long-term
    cache = get_cache_manager()
    hist_str = format_history(history, max_items=1) # Just a hint
    rewrite_cache_key = f"rewrite_cache:{hashlib.md5(f'{question}:{hist_str}'.encode()).hexdigest()}"
    
    cached_rewrite = cache.get(rewrite_cache_key)
    if cached_rewrite:
        return cached_rewrite

    history_text = format_history(history)
    prompt = QUERY_REWRITE_PROMPT.format(
        history=history_text,
        question=question,
    )
    res = _get_llm().invoke(prompt).content.strip()
    
    # Fallback
    if len(res) < 2:
        res = question
        
    # Cache the rewrite result for 1 hour
    cache.set(rewrite_cache_key, res, ttl=3600)
    return res


async def rewrite_query_async(query: str, history: list[str]) -> str:
    """Rewrite query to make it standalone using context asynchronously with caching."""
    cache = get_cache_manager()
    hist_str = format_history(history, max_items=1)
    rewrite_cache_key = f"rewrite_cache:{hashlib.md5(f'{query}:{hist_str}'.encode()).hexdigest()}"
    
    cached_rewrite = cache.get(rewrite_cache_key)
    if cached_rewrite:
        return cached_rewrite

    history_text = format_history(history)
    prompt = QUERY_REWRITE_PROMPT.format(question=query, history=history_text)

    try:
        llm = _get_llm()
        response = await llm.ainvoke(prompt)
        content = response.content
        if isinstance(content, str):
            rewritten = content.strip()
            res = rewritten if rewritten and len(rewritten) > 1 else query
            cache.set(rewrite_cache_key, res, ttl=3600)
            return res
        return query
    except Exception as e:
        logger.error(f"Query rewrite failed: {e}")
        return query


def rag_pipeline(question: str, history: Any) -> dict[str, Any]:
    """Run the RAG pipeline with two-layer caching."""
    total_start = time.monotonic()
    cache = get_cache_manager()
    quick_key = _get_cache_key(question, history)
    
    logger.info(f"Checking RAG Quick Cache: key={quick_key}")
    cached = cache.get(quick_key)
    if cached:
        logger.info(f"RAG Quick Cache HIT for: {question[:50]}")
        try:
            log_query(
                question=question,
                answer=cached["answer"],
                response_time=0.01,
                retrieval_time=0,
                generation_time=0,
                docs_retrieved=cached.get("metrics", {}).get("docs_retrieved", 0),
                chunks_processed=cached.get("metrics", {}).get("chunks_processed", 0),
                retrieval_scores=cached.get("metrics", {}).get("retrieval_scores", []),
                thoughts=cached.get("thoughts", ""),
                standalone_query=question
            )
        except Exception:
            logger.warning("Failed to log cached query metrics", exc_info=True)
        return cached

    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            standalone_query = rewrite_query(question, history)
            logger.info(f"Standalone Query: {standalone_query}")
            
            # --- SEMANTIC CACHE CHECK ---
            # Check for semantically similar queries to avoid redundant RAG processing.
            embedder = get_embeddings()
            query_embedding = embedder.embed_query(standalone_query)
            
            semantic_key = cache.get_semantic(query_embedding, threshold=0.98)
            if semantic_key:
                cached_sem = cache.get(semantic_key)
                if cached_sem:
                    logger.info(f"RAG Semantic Cache HIT for: {standalone_query[:50]}")
                    try:
                        log_query(
                            question=question,
                            answer=cached_sem["answer"],
                            response_time=round(time.monotonic() - total_start, 2),
                            retrieval_time=0,
                            generation_time=0,
                            docs_retrieved=cached_sem.get("metrics", {}).get("docs_retrieved", 0),
                            chunks_processed=cached_sem.get("metrics", {}).get("chunks_processed", 0),
                            retrieval_scores=cached_sem.get("metrics", {}).get("retrieval_scores", []),
                            thoughts=cached_sem.get("thoughts", ""),
                            standalone_query=standalone_query
                        )
                    except Exception:
                        logger.warning("Failed to log semantic cached query metrics", exc_info=True)
                    cache.set(quick_key, cached_sem, ttl=3600)
                    return cached_sem
            # ----------------------------
            
            standalone_normalized = standalone_query.strip().lower()
            standalone_hash = hashlib.md5(standalone_normalized.encode()).hexdigest()
            deep_key = f"rag_deep_cache:{standalone_hash}"
            
            logger.info(f"Checking RAG Deep Cache: query='{standalone_query}', key={deep_key}")
            cached_deep = cache.get(deep_key)
            if cached_deep:
                logger.info(f"RAG Deep Cache HIT for: {standalone_query[:50]}")
                try:
                    log_query(
                        question=question,
                        answer=cached_deep["answer"],
                        response_time=round(time.monotonic() - total_start, 2),
                        retrieval_time=0,
                        generation_time=0,
                        docs_retrieved=cached_deep.get("metrics", {}).get("docs_retrieved", 0),
                        chunks_processed=cached_deep.get("metrics", {}).get("chunks_processed", 0),
                        retrieval_scores=cached_deep.get("metrics", {}).get("retrieval_scores", []),
                        thoughts=cached_deep.get("thoughts", ""),
                        standalone_query=standalone_query
                    )
                except Exception:
                    logger.warning("Failed to log deep cached query metrics", exc_info=True)
                cache.set(quick_key, cached_deep, ttl=3600)
                return cached_deep

            result = _run_rag_pipeline_with_query(standalone_query, question, history)
            cache.set(deep_key, result, ttl=3600)
            cache.set(quick_key, result, ttl=3600)
            # Store in semantic cache
            cache.add_semantic(query_embedding, deep_key)
            return result
        except Exception as exc:
            if attempt < (max_attempts - 1) and is_qdrant_client_closed_error(exc):
                logger.warning("Qdrant client was closed during query; recreating and retrying once")
                close_qdrant_client()
                continue
            raise
    raise RuntimeError("Unexpected RAG retry state")


def _run_rag_pipeline_with_query(standalone_query: str, original_question: str, history: Any) -> dict[str, Any]:
    """Internal helper for core RAG steps."""
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

    context, sources_str, sources_json = format_docs(reranked_docs)
    prompt = RAG_PROMPT.format(context=context, question=original_question, sources=sources_str)

    generation_start = time.monotonic()
    
    # Handle thinking in non-streaming mode
    parser = ThinkingParser()
    raw_answer = _get_llm().invoke(prompt).content
    full_answer = ""
    full_thoughts = ""
    for msg_type, content in parser.feed(raw_answer):
        if msg_type == "thinking":
            full_thoughts += content
        else:
            full_answer += content
    
    generation_time = time.monotonic() - generation_start
    total_time = time.monotonic() - total_start

    res = {
        "answer": full_answer,
        "thoughts": full_thoughts,
        "sources": sources_json,
        "metrics": {
            "total_time": round(total_time, 2),
            "retrieval_time": round(retrieval_time, 2),
            "generation_time": round(generation_time, 2),
            "docs_retrieved": len(retrieved_docs),
            "chunks_processed": len(reranked_docs),
            "retrieval_scores": retrieval_scores,
        },
    }

    try:
        log_query(
            question=original_question,
            response_time=total_time,
            answer_length=len(full_answer),
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            docs_retrieved=len(retrieved_docs),
            chunks_processed=len(reranked_docs),
            retrieval_scores=retrieval_scores,
            thoughts=full_thoughts,
            standalone_query=standalone_query,
        )
    except Exception:
        logger.warning("Failed to log query metrics", exc_info=True)

    return res


async def rag_pipeline_stream_async(query: str, chat_history: list[str]):
    total_start = time.monotonic()
    cache = get_cache_manager()
    quick_key = _get_cache_key(query, chat_history)

    logger.info(f"Stream: Checking Quick Cache: key={quick_key}")
    cached = cache.get(quick_key)
    if cached:
        logger.info("Stream: Quick Cache HIT")
        yield json.dumps({"step": "cached", "answer": cached["answer"]}) + "\n"
        yield json.dumps({
            "step": "done",
            "answer": cached["answer"],
            "thoughts": cached.get("thoughts", ""),
            "sources": cached["sources"],
            "metrics": cached.get("metrics", {})
        }) + "\n"
        
        try:
            log_query(
                question=query,
                response_time=0.01,
                answer_length=len(cached.get("answer", "")),
                retrieval_time=0,
                generation_time=0,
                docs_retrieved=cached.get("metrics", {}).get("docs_retrieved", 0),
                chunks_processed=cached.get("metrics", {}).get("chunks_processed", 0),
                retrieval_scores=cached.get("metrics", {}).get("retrieval_scores", []),
                thoughts=cached.get("thoughts", ""),
                standalone_query=query,
            )
        except Exception:
            pass
        return

    try:
        yield json.dumps({"step": "rewriting"}) + "\n"
        standalone_query = await rewrite_query_async(query, chat_history)
        logger.info(f"Stream Standalone Query: {standalone_query}")
        
        # --- SEMANTIC CACHE CHECK ---
        embedder = get_embeddings()
        query_embedding = embedder.embed_query(standalone_query)
        
        semantic_key = cache.get_semantic(query_embedding, threshold=0.98)
        if semantic_key:
            cached_sem = cache.get(semantic_key)
            if cached_sem:
                logger.info("Stream: Semantic Cache HIT")
                yield json.dumps({"step": "cached", "answer": cached_sem["answer"]}) + "\n"
                yield json.dumps({
                    "step": "done",
                    "answer": cached_sem["answer"],
                    "thoughts": cached_sem.get("thoughts", ""),
                    "sources": cached_sem["sources"],
                    "metrics": cached_sem.get("metrics", {})
                }) + "\n"
                
                try:
                    log_query(
                        question=query,
                        response_time=time.monotonic() - total_start,
                        answer_length=len(cached_sem["answer"]),
                        retrieval_time=0,
                        generation_time=0,
                        thoughts=cached_sem.get("thoughts", ""),
                        standalone_query=standalone_query,
                    )
                except Exception:
                    pass
                cache.set(quick_key, cached_sem, ttl=3600)
                return
        # ----------------------------
        
        standalone_normalized = standalone_query.strip().lower()
        standalone_hash = hashlib.md5(standalone_normalized.encode()).hexdigest()
        deep_key = f"rag_deep_cache:{standalone_hash}"
        logger.info(f"Stream: Checking Deep Cache: query='{standalone_query}', key={deep_key}")
        cached_deep = cache.get(deep_key)
        if cached_deep:
            logger.info("Stream: Deep Cache HIT")
            yield json.dumps({"step": "cached"}) + "\n"
            yield json.dumps({
                "step": "done",
                "answer": cached_deep["answer"],
                "thoughts": cached_deep.get("thoughts", ""),
                "sources": cached_deep["sources"],
                "metrics": cached_deep.get("metrics", {})
            }) + "\n"
            
            try:
                log_query(
                    question=query,
                    response_time=time.monotonic() - total_start,
                    answer_length=len(cached_deep["answer"]),
                    retrieval_time=0,
                    generation_time=0,
                    thoughts=cached_deep.get("thoughts", ""),
                    standalone_query=standalone_query,
                )
            except Exception:
                pass
            return

        yield json.dumps({"step": "retrieving", "query": standalone_query}) + "\n"
        retriever, vectorstore, client = build_retriever()
        
        retrieval_start = time.monotonic()
        try:
            docs, metadata = await retrieve_documents_async(standalone_query, dense_retriever=retriever, client=client)
        except Exception as e:
            if is_qdrant_client_closed_error(e):
                close_qdrant_client()
                retriever, vectorstore, client = build_retriever()
                docs, metadata = await retrieve_documents_async(standalone_query, dense_retriever=retriever, client=client)
            else:
                raise
        retrieval_time = time.monotonic() - retrieval_start

        if not docs:
            yield json.dumps({
                "step": "done",
                "answer": "Maaf, tidak ditemukan informasi relevan.",
                "sources": [],
                "metrics": metadata
            }) + "\n"
            return

        yield json.dumps({"step": "reranking"}) + "\n"
        top_docs, retrieval_scores = rerank(standalone_query, docs)
        metadata["retrieval_scores"] = retrieval_scores
        metadata["reranked_docs"] = len(top_docs)

        yield json.dumps({"step": "generating"}) + "\n"
        context_text, sources_str, sources_json = format_docs(top_docs)
        prompt = RAG_PROMPT.format(context=context_text, question=standalone_query, sources=sources_str)
        
        llm = _get_llm()
        generation_start = time.monotonic()
        parser = ThinkingParser()
        full_answer = ""
        full_thoughts = ""
        
        async for chunk in llm.astream(prompt):
            if chunk.content:
                for msg_type, content in parser.feed(chunk.content):
                    if msg_type == "thinking":
                        full_thoughts += content
                        yield json.dumps({"step": "thinking", "content": content}) + "\n"
                    else:
                        full_answer += content
                        yield json.dumps({"step": "token", "content": content}) + "\n"
        
        # Flush remaining buffer
        for msg_type, content in parser.flush():
            if msg_type == "thinking":
                full_thoughts += content
                yield json.dumps({"step": "thinking", "content": content}) + "\n"
            else:
                full_answer += content
                yield json.dumps({"step": "token", "content": content}) + "\n"
        
        generation_time = time.monotonic() - generation_start
        total_time = time.monotonic() - total_start

        final_result = {
            "step": "done",
            "answer": full_answer,
            "thoughts": full_thoughts,
            "sources": sources_json,
            "metrics": {
                **metadata,
                "total_time": round(total_time, 2),
                "retrieval_time": round(retrieval_time, 2),
                "generation_time": round(generation_time, 2),
            }
        }
        
        cache_data = {
            "answer": full_answer,
            "thoughts": full_thoughts,
            "sources": sources_json,
            "metrics": final_result["metrics"]
        }
        cache.set(deep_key, cache_data, ttl=3600)
        cache.set(quick_key, cache_data, ttl=3600)
        # Store in semantic cache
        cache.add_semantic(query_embedding, deep_key)
        
        yield json.dumps(final_result) + "\n"

        try:
            log_query(
                question=query,
                response_time=total_time,
                answer_length=len(full_answer),
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                docs_retrieved=len(docs),
                chunks_processed=len(top_docs),
                retrieval_scores=metadata.get("retrieval_scores", []),
                thoughts=full_thoughts,
                standalone_query=standalone_query,
            )
        except Exception:
            pass

    except Exception as e:
        logger.exception("Pipeline error")
        yield json.dumps({"step": "error", "message": str(e)}) + "\n"
