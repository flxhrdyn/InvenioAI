"""Cross-encoder reranking.

Reranking is best-effort: if the reranker cannot be loaded (missing wheels,
network restrictions, HF rate limits), the app falls back to the base retriever
order instead of failing the whole request.
"""

from __future__ import annotations

import logging
from typing import Any, List, Tuple

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


def rerank(query: str, docs: List[Any]) -> Tuple[List[Any], List[float]]:
    """Return documents ordered by cross-encoder relevance.

    Args:
        query: User query (or rewritten query).
        docs: Retrieved documents (LangChain `Document`-like objects).

    Returns:
        Tuple of (ranked_docs, scores). Up to `RERANK_TOP_K` items.
    """

    if not docs:
        return [], []

    try:
        # Convert LangChain documents to FlashRank format
        passages = [
            {"id": i, "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(docs)
        ]
        
        rerank_request = RerankRequest(query=query, passages=passages)
        results = _get_ranker().rerank(rerank_request)
        
        # FlashRank results are already sorted by score
        ranked_docs = []
        scores = []
        for res in results[:RERANK_TOP_K]:
            doc_id = res["id"]
            ranked_docs.append(docs[doc_id])
            scores.append(round(float(res.get("score", 0.0)), 4))
            
        return ranked_docs, scores
        
    except Exception as exc:
        logger.warning("Reranker unavailable; skipping rerank (%s)", exc)
        # Fallback: original order, zero scores
        top_docs = docs[:RERANK_TOP_K]
        return top_docs, [0.0] * len(top_docs)