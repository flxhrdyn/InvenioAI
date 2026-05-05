"""Cross-encoder reranking.

Reranking is best-effort: if the reranker cannot be loaded (missing wheels,
network restrictions, HF rate limits), the app falls back to the base retriever
order instead of failing the whole request.
"""

from __future__ import annotations

import logging
from typing import Any, List

# Import config FIRST so Hugging Face Hub env defaults (timeouts/offline) are set
# before importing the HF stack.
from . import config as _config  # noqa: F401
from .config import RERANKER_MODEL, RERANK_TOP_K

from flashrank import Ranker, RerankRequest

logger = logging.getLogger(__name__)

_ranker: Ranker | None = None


def _get_ranker() -> Ranker:
    global _ranker
    if _ranker is None:
        # FlashRank uses ONNX for high-performance CPU inference
        _ranker = Ranker(model_name=RERANKER_MODEL)
    return _ranker


def rerank(query: str, docs: List[Any]) -> List[Any]:
    """Return documents ordered by cross-encoder relevance.

    Args:
        query: User query (or rewritten query).
        docs: Retrieved documents (LangChain `Document`-like objects).

    Returns:
        Up to `RERANK_TOP_K` documents. If reranking fails, returns the first
        `RERANK_TOP_K` documents in their original order.
    """

    if not docs:
        return []

    try:
        # Convert LangChain documents to FlashRank format
        passages = [
            {"id": i, "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(docs)
        ]
        
        rerank_request = RerankRequest(query=query, passages=passages)
        results = _get_ranker().rerank(rerank_request)
        
        # FlashRank results are already sorted by score
        # We need to map them back to the original document objects
        ranked_docs = []
        for res in results[:RERANK_TOP_K]:
            doc_id = res["id"]
            ranked_docs.append(docs[doc_id])
            
        return ranked_docs
        
    except Exception as exc:
        # Reranking is an optimization; treat failures as non-fatal.
        logger.warning("Reranker unavailable; skipping rerank (%s)", exc)
        return docs[:RERANK_TOP_K]