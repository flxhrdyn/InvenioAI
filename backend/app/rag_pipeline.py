"""RAG pipeline orchestration.

The end-to-end flow is:

rewrite query → retrieve (dense or hybrid) → rerank → generate answer → return metrics.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
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


def format_history(history: Any) -> str:
    """Format chat history into a prompt-friendly string."""
    if not history:
        return ""

    if isinstance(history, str):
        return history

    return "\n".join(str(item) for item in history)


def rewrite_query(question: str, history: Any) -> str:
    """Rewrite a question into a standalone query."""
    prompt = QUERY_REWRITE_PROMPT.format(
        history=format_history(history),
        question=question,
    )
    return _get_llm().invoke(prompt).content


def _run_rag_pipeline_once(question: str, history: Any) -> dict[str, Any]:
    total_start = time.monotonic()
    retriever, vectorstore, client = build_retriever()

    try:
        standalone_query = rewrite_query(question, history)

        retrieval_start = time.monotonic()
        retrieved_docs, retrieval_meta = retrieve_documents(
            standalone_query,
            dense_retriever=retriever,
            client=client,
        )

        reranked_docs = rerank(standalone_query, retrieved_docs)
        retrieval_time = time.monotonic() - retrieval_start

        # Per-document similarity scores for the dashboard.
        try:
            scored = vectorstore.similarity_search_with_relevance_scores(
                standalone_query, k=RETRIEVAL_K
            )
            retrieval_scores = [round(float(score), 4) for _, score in scored]
        except Exception:
            logger.debug("Failed to fetch relevance scores", exc_info=True)
            retrieval_scores = []

        context, sources_str, sources_json = format_docs(reranked_docs)
        prompt = RAG_PROMPT.format(context=context, question=question, sources=sources_str)

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
        log_query(
            question=question,
            response_time=total_time,
            answer_length=len(res["answer"]),
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            docs_retrieved=len(retrieved_docs),
            chunks_processed=len(reranked_docs),
            retrieval_scores=retrieval_scores,
        )

        return res
    finally:
        # Qdrant client is managed as a process-wide singleton and closed on
        # application shutdown.
        pass


async def rewrite_query_async(query: str, history: list[str]) -> str:
    """Rewrite query to make it standalone using context asynchronously."""
    if not history:
        return query

    context = "\n".join([f"- {msg}" for msg in history[-3:]])
    prompt = QUERY_REWRITE_PROMPT.format(query=query, context=context)

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
    """Run the RAG pipeline.

    Returns a dict containing the answer, a sources string, and timing metrics.
    """

    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            return _run_rag_pipeline_once(question, history)
        except Exception as exc:
            if attempt < (max_attempts - 1) and is_qdrant_client_closed_error(exc):
                logger.warning("Qdrant client was closed during query; recreating and retrying once")
                close_qdrant_client()
                continue
            raise

    raise RuntimeError("Unexpected RAG retry state")

async def rag_pipeline_stream_async(query: str, chat_history: list[str]):
    from .retriever import build_retriever, retrieve_documents_async
    from .qdrant_conn import is_qdrant_client_closed_error, close_qdrant_client
    from .reranker import rerank
    from .config import RETRIEVAL_K
    import time

    total_start = time.monotonic()

    try:
        # Step 1: Rewrite Query asynchronously
        yield json.dumps({"step": "rewriting"}) + "\n"
        standalone_query = await rewrite_query_async(query, chat_history)

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
            yield json.dumps({
                "step": "done",
                "answer": "Maaf, saya tidak menemukan informasi relevan mengenai pertanyaan tersebut di dalam dokumen.",
                "docs": [],
                "metadata": metadata
            }) + "\n"
            return

        # Step 3: Reranking
        yield json.dumps({"step": "reranking"}) + "\n"
        top_docs = rerank(standalone_query, docs, top_k=RETRIEVAL_K)
        metadata["reranked_docs"] = len(top_docs)

        # Step 4: LLM Answer Generation (Streaming)
        yield json.dumps({"step": "generating"}) + "\n"
        
        context_text = "\n\n---\n\n".join(
            [f"[Sumber: {d.metadata.get('source_file', 'unknown')}]\n{d.page_content}" for d in top_docs]
        )
        prompt = RAG_PROMPT.format(context=context_text, question=standalone_query, sources="")
        
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
        yield json.dumps({
            "step": "done",
            "answer": full_answer,
            "docs": [d.model_dump() if hasattr(d, "model_dump") else str(d) for d in top_docs],
            "metadata": {
                **metadata,
                "total_time": round(total_time, 2),
                "retrieval_time": round(retrieval_time, 2),
                "generation_time": round(generation_time, 2),
            }
        }) + "\n"

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
