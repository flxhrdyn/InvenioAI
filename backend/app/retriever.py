"""Retriever construction.

All retrieval wiring lives here (embeddings, Qdrant vector store, retriever
strategy). The RAG pipeline can then focus on orchestration and metrics.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any, Dict, List
from typing import Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from .embeddings import get_embeddings
from .config import (
    GROQ_API_KEY,
    HYBRID_DENSE_WEIGHT,
    HYBRID_FUSION_LIMIT,
    HYBRID_LEXICAL_K,
    HYBRID_LEXICAL_WEIGHT,
    HYBRID_MAX_LEXICAL_DOCS,
    HYBRID_RRF_K,
    LLM_MODEL,
    QDRANT_COLLECTION,
    RETRIEVAL_K,
    USE_HYBRID_SEARCH,
)
from .qdrant_conn import get_qdrant_client


logger = logging.getLogger(__name__)

# Default to quieter logs; let the application configure logging if needed.
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.WARNING)

_bm25_cache_lock = threading.Lock()
_bm25_cache: BM25Retriever | None = None
_bm25_cache_count: int | None = None


def _doc_key(doc: Document) -> str:
    metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
    source = str(metadata.get("source", ""))
    page = str(metadata.get("page", ""))
    chunk_id = str(metadata.get("chunk_id", ""))
    content_hash = hashlib.sha1(doc.page_content.encode("utf-8", errors="ignore")).hexdigest()
    return f"{source}|{page}|{chunk_id}|{content_hash}"


def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    *,
    rrf_k: int,
    weights: List[float],
    max_results: int,
) -> List[Document]:
    """Fuse multiple ranked lists using weighted Reciprocal Rank Fusion."""

    if not ranked_lists:
        return []

    scores: Dict[str, float] = {}
    docs_by_key: Dict[str, Document] = {}

    for list_index, ranked_docs in enumerate(ranked_lists):
        weight = weights[list_index] if list_index < len(weights) else 1.0
        if weight <= 0:
            continue
        for rank, doc in enumerate(ranked_docs, start=1):
            key = _doc_key(doc)
            docs_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + (weight / (rrf_k + rank))

    fused_keys = sorted(scores, key=scores.get, reverse=True)
    if max_results > 0:
        fused_keys = fused_keys[:max_results]
    return [docs_by_key[key] for key in fused_keys]


def _payload_to_document(payload: Dict[str, Any]) -> Document | None:
    content = payload.get("page_content") or payload.get("text") or payload.get("document")
    if not isinstance(content, str) or not content.strip():
        return None

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    metadata = dict(metadata)
    if "source" in payload and "source" not in metadata:
        metadata["source"] = payload["source"]

    return Document(page_content=content, metadata=metadata)


def _collection_points_count(client: QdrantClient) -> int:
    return int(client.count(collection_name=QDRANT_COLLECTION, exact=False).count or 0)


def _load_documents_for_bm25(client: QdrantClient, max_docs: int) -> List[Document]:
    documents: List[Document] = []
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=next_offset,
        )

        if not points:
            break

        for point in points:
            payload = getattr(point, "payload", None) or {}
            if not isinstance(payload, dict):
                continue
            doc = _payload_to_document(payload)
            if doc is not None:
                documents.append(doc)
            if len(documents) >= max_docs:
                return documents

        if next_offset is None:
            break

    return documents


def _get_bm25_retriever(client: QdrantClient) -> BM25Retriever | None:
    """Return cached BM25 retriever and rebuild when collection size changes."""

    global _bm25_cache, _bm25_cache_count
    try:
        point_count = _collection_points_count(client)
    except Exception:
        logger.warning("Could not read Qdrant point count for BM25 cache", exc_info=True)
        return None

    if _bm25_cache is not None and _bm25_cache_count == point_count:
        return _bm25_cache

    with _bm25_cache_lock:
        if _bm25_cache is not None and _bm25_cache_count == point_count:
            return _bm25_cache

        try:
            docs = _load_documents_for_bm25(client, max_docs=HYBRID_MAX_LEXICAL_DOCS)
        except Exception:
            logger.warning("Failed to load documents for BM25 retriever", exc_info=True)
            return None

        if not docs:
            return None

        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = HYBRID_LEXICAL_K

        _bm25_cache = bm25
        _bm25_cache_count = point_count
        logger.debug("BM25 retriever rebuilt (docs=%s)", len(docs))
        return _bm25_cache


def build_retriever() -> Tuple[MultiQueryRetriever, QdrantVectorStore, QdrantClient]:
    """Build and validate the retriever stack.

    Returns a tuple of (retriever, vectorstore, client). Raises a ValueError if
    the API key is missing or the expected Qdrant collection does not exist.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY belum di-set. Isi di .env (lihat .env.example) sebelum menjalankan query."
        )

    # Embedding model (cached)
    embeddings = get_embeddings()

    # Shared Qdrant client (avoids local storage lock issues)
    client = get_qdrant_client()

    # Guard: ensure the collection exists before serving queries.
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        raise ValueError(
            "Belum ada dokumen yang diindex. Upload dan index PDF terlebih dahulu."
        )

    # Vector store wrapper
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )

    # Base retriever with MMR for diversity
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVAL_K,
            "fetch_k": max(RETRIEVAL_K * 4, RETRIEVAL_K),
            "lambda_mult": 0.5
        }
    )

    # Groq LLM
    llm = ChatGroq(
        model=LLM_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0
    )

    # MultiQuery Retriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

    logger.debug("Retriever ready (collection=%s, k=%s)", QDRANT_COLLECTION, RETRIEVAL_K)
    return retriever, vectorstore, client


def retrieve_documents(
    query: str,
    *,
    dense_retriever: MultiQueryRetriever,
    client: QdrantClient,
) -> Tuple[List[Document], Dict[str, Any]]:
    """Retrieve documents with dense-only or hybrid strategy.

    Hybrid mode combines:
    - dense branch: MultiQueryRetriever + MMR
    - lexical branch: BM25 over indexed chunks
    - fusion: weighted Reciprocal Rank Fusion (RRF)
    """

    dense_docs = dense_retriever.invoke(query)
    metadata: Dict[str, Any] = {
        "mode": "dense",
        "dense_docs": len(dense_docs),
        "lexical_docs": 0,
        "fused_docs": len(dense_docs),
    }

    if not USE_HYBRID_SEARCH:
        return dense_docs, metadata

    bm25_retriever = _get_bm25_retriever(client)
    if bm25_retriever is None:
        metadata["mode"] = "dense-fallback"
        return dense_docs, metadata

    try:
        lexical_docs = bm25_retriever.invoke(query)
    except Exception:
        logger.warning("Lexical retrieval failed; fallback to dense branch", exc_info=True)
        metadata["mode"] = "dense-fallback"
        return dense_docs, metadata

    fused_docs = reciprocal_rank_fusion(
        [dense_docs, lexical_docs],
        rrf_k=HYBRID_RRF_K,
        weights=[HYBRID_DENSE_WEIGHT, HYBRID_LEXICAL_WEIGHT],
        max_results=HYBRID_FUSION_LIMIT,
    )
    if not fused_docs:
        metadata["mode"] = "dense-fallback"
        return dense_docs, metadata

    metadata.update(
        {
            "mode": "hybrid",
            "lexical_docs": len(lexical_docs),
            "fused_docs": len(fused_docs),
        }
    )
    return fused_docs, metadata

import asyncio

async def retrieve_documents_async(
    query: str,
    dense_retriever: MultiQueryRetriever,
    client: QdrantClient,
) -> Tuple[List[Document], Dict[str, Any]]:
    """Retrieve documents asynchronously with dense-only or hybrid strategy."""
    
    # Run dense retrieval asynchronously
    dense_docs = await dense_retriever.ainvoke(query)
    
    metadata: Dict[str, Any] = {
        "mode": "dense",
        "dense_docs": len(dense_docs),
        "lexical_docs": 0,
        "fused_docs": len(dense_docs),
    }

    if not USE_HYBRID_SEARCH:
        return dense_docs, metadata

    # Getting BM25 retriever is sync, but it's cached or fast enough
    bm25_retriever = _get_bm25_retriever(client)
    if bm25_retriever is None:
        metadata["mode"] = "dense-fallback"
        return dense_docs, metadata

    try:
        # Run lexical retrieval (ainvoke if supported, else fallback to invoke in a thread)
        if hasattr(bm25_retriever, "ainvoke"):
            lexical_docs = await bm25_retriever.ainvoke(query)
        else:
            lexical_docs = await asyncio.to_thread(bm25_retriever.invoke, query)
    except Exception:
        logger.warning("Lexical retrieval failed; fallback to dense branch", exc_info=True)
        metadata["mode"] = "dense-fallback"
        return dense_docs, metadata

    fused_docs = reciprocal_rank_fusion(
        [dense_docs, lexical_docs],
        rrf_k=HYBRID_RRF_K,
        weights=[HYBRID_DENSE_WEIGHT, HYBRID_LEXICAL_WEIGHT],
        max_results=HYBRID_FUSION_LIMIT,
    )
    
    if not fused_docs:
        metadata["mode"] = "dense-fallback"
        return dense_docs, metadata

    metadata.update(
        {
            "mode": "hybrid",
            "lexical_docs": len(lexical_docs),
            "fused_docs": len(fused_docs),
        }
    )
    return fused_docs, metadata