"""Local metrics store used by the Streamlit dashboard.

Metrics are stored in a small JSON file (`metrics.json`) so the dashboard can
render without an external database.
"""
import logging
import json
import math
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import METRICS_FILE


logger = logging.getLogger(__name__)


def load_metrics() -> Dict[str, Any]:
    """Load metrics from disk.

    Returns default values when the file is missing or unreadable.
    """
    default_metrics = {
        "total_queries": 0,
        "total_documents_indexed": 0,
        "total_response_time": 0,
        "total_retrieval_time": 0,
        "total_generation_time": 0,
        "total_docs_retrieved": 0,
        "total_chunks_processed": 0,
        "query_history": []
    }
    
    if not os.path.exists(METRICS_FILE):
        return default_metrics
    
    try:
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            # Merge with default to ensure all keys exist (backward compatibility)
            for key in default_metrics:
                if key not in metrics:
                    metrics[key] = default_metrics[key]
            return metrics
    except Exception:
        logger.exception("Failed to load metrics file: %s", METRICS_FILE)
        return default_metrics


def save_metrics(metrics: Dict[str, Any]) -> None:
    """Save metrics to disk."""
    try:
        parent_dir = os.path.dirname(METRICS_FILE)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Atomic write to avoid corrupting the JSON file on crashes or
        # concurrent writes.
        fd, tmp_path = tempfile.mkstemp(prefix="metrics_", suffix=".json", dir=parent_dir or None)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            os.replace(tmp_path, METRICS_FILE)
        finally:
            # If os.replace fails, best-effort cleanup.
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
    except Exception:
        logger.exception("Failed to save metrics file: %s", METRICS_FILE)


def log_query(
    question: str, 
    response_time: float, 
    answer_length: int,
    retrieval_time: float = 0,
    generation_time: float = 0,
    docs_retrieved: int = 0,
    chunks_processed: int = 0,
    retrieval_scores: Optional[List[float]] = None,
) -> None:
    """Log a query with RAG metrics."""
    metrics = load_metrics()
    
    metrics["total_queries"] += 1
    metrics["total_response_time"] += response_time
    metrics["total_retrieval_time"] += retrieval_time
    metrics["total_generation_time"] += generation_time
    metrics["total_docs_retrieved"] += docs_retrieved
    metrics["total_chunks_processed"] += chunks_processed
    
    # Keep only last 100 queries to prevent file from growing too large
    if len(metrics["query_history"]) >= 100:
        metrics["query_history"].pop(0)
    
    metrics["query_history"].append({
        "timestamp": datetime.now().isoformat(),
        "question": question[:100],  # Truncate long questions
        "response_time": round(response_time, 2),
        "retrieval_time": round(retrieval_time, 2),
        "generation_time": round(generation_time, 2),
        "answer_length": answer_length,
        "docs_retrieved": docs_retrieved,
        "chunks_processed": chunks_processed,
        "retrieval_scores": retrieval_scores or [],
    })
    
    save_metrics(metrics)


def log_document_indexed() -> None:
    """Record that a document was indexed."""
    metrics = load_metrics()
    metrics["total_documents_indexed"] += 1
    save_metrics(metrics)


def get_avg_response_time() -> float:
    """Return average total response time."""
    metrics = load_metrics()
    if metrics["total_queries"] == 0:
        return 0.0
    return round(metrics["total_response_time"] / metrics["total_queries"], 2)


def get_avg_retrieval_time() -> float:
    """Return average retrieval time."""
    metrics = load_metrics()
    if metrics["total_queries"] == 0:
        return 0.0
    return round(metrics["total_retrieval_time"] / metrics["total_queries"], 2)


def get_avg_generation_time() -> float:
    """Return average generation time."""
    metrics = load_metrics()
    if metrics["total_queries"] == 0:
        return 0.0
    return round(metrics["total_generation_time"] / metrics["total_queries"], 2)


def get_avg_docs_retrieved() -> float:
    """Return average number of retrieved documents per query."""
    metrics = load_metrics()
    if metrics["total_queries"] == 0:
        return 0.0
    return round(metrics["total_docs_retrieved"] / metrics["total_queries"], 1)


def get_avg_chunks_processed() -> float:
    """Return average number of chunks processed per query."""
    metrics = load_metrics()
    if metrics["total_queries"] == 0:
        return 0.0
    return round(metrics["total_chunks_processed"] / metrics["total_queries"], 1)


def get_retrieval_efficiency() -> float:
    """Return retrieval time as a percentage of total time."""
    metrics = load_metrics()
    if metrics["total_response_time"] == 0:
        return 0.0
    return round((metrics["total_retrieval_time"] / metrics["total_response_time"]) * 100, 1)


def get_generation_efficiency() -> float:
    """Return generation time as a percentage of total time."""
    metrics = load_metrics()
    if metrics["total_response_time"] == 0:
        return 0.0
    return round((metrics["total_generation_time"] / metrics["total_response_time"]) * 100, 1)


def reset_metrics() -> None:
    """Reset all stored metrics."""
    metrics = {
        "total_queries": 0,
        "total_documents_indexed": 0,
        "total_response_time": 0,
        "total_retrieval_time": 0,
        "total_generation_time": 0,
        "total_docs_retrieved": 0,
        "total_chunks_processed": 0,
        "query_history": []
    }
    save_metrics(metrics)


# ── IR Metric Computation ─────────────────────────────────────────────────────

def _binary_relevance(scores: List[float], threshold: float) -> List[int]:
    """Return binary relevance list: 1 if score >= threshold, else 0."""
    return [1 if s >= threshold else 0 for s in scores]


def precision_at_k(scores: List[float], k: int, threshold: float = 0.5) -> float:
    """Precision@k: fraction of top-k retrieved docs that are relevant."""
    if not scores or k == 0:
        return 0.0
    rel = _binary_relevance(scores[:k], threshold)
    return sum(rel) / k


def recall_at_k(scores: List[float], k: int, threshold: float = 0.5) -> float:
    """Recall@k: fraction of relevant docs found in top-k.

    Since total relevant in the collection is unknown, we approximate using the
    total relevant among all retrieved scores as a lower bound.
    """
    if not scores:
        return 0.0
    total_relevant = sum(_binary_relevance(scores, threshold))
    if total_relevant == 0:
        return 0.0
    relevant_in_k = sum(_binary_relevance(scores[:k], threshold))
    return relevant_in_k / total_relevant


def mrr(scores: List[float], threshold: float = 0.5) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant document."""
    for i, s in enumerate(scores):
        if s >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(scores: List[float], k: int) -> float:
    """nDCG@k: uses raw similarity scores as graded relevance."""
    if not scores:
        return 0.0
    top_k = scores[:k]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(top_k))
    ideal = sorted(scores, reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return (dcg / idcg) if idcg > 0 else 0.0


def hit_rate_at_k(scores: List[float], k: int, threshold: float = 0.5) -> float:
    """HitRate@k: 1 if at least one relevant doc in top-k, else 0."""
    if not scores:
        return 0.0
    return 1.0 if any(s >= threshold for s in scores[:k]) else 0.0


def compute_ir_metrics(
    query_history: List[Dict[str, Any]],
    k: int = 5,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute aggregate IR metrics from stored query history."""
    entries_with_scores = [
        q for q in query_history if q.get("retrieval_scores")
    ]
    if not entries_with_scores:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "hit_rate": 0.0,
            "evaluated_queries": 0,
        }

    p_vals, r_vals, mrr_vals, ndcg_vals, hr_vals = [], [], [], [], []
    for q in entries_with_scores:
        s = q["retrieval_scores"]
        p_vals.append(precision_at_k(s, k, threshold))
        r_vals.append(recall_at_k(s, k, threshold))
        mrr_vals.append(mrr(s, threshold))
        ndcg_vals.append(ndcg_at_k(s, k))
        hr_vals.append(hit_rate_at_k(s, k, threshold))

    n = len(entries_with_scores)
    return {
        "precision": round(sum(p_vals) / n, 4),
        "recall": round(sum(r_vals) / n, 4),
        "mrr": round(sum(mrr_vals) / n, 4),
        "ndcg": round(sum(ndcg_vals) / n, 4),
        "hit_rate": round(sum(hr_vals) / n, 4),
        "evaluated_queries": n,
    }


def per_query_ir_metrics(
    query_history: List[Dict[str, Any]],
    k: int = 5,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Return per-query IR metrics for charting."""
    rows = []
    for q in query_history:
        s = q.get("retrieval_scores", [])
        rows.append({
            "timestamp": q.get("timestamp", ""),
            "question": q.get("question", "")[:50],
            "response_time": q.get("response_time", 0),
            "retrieval_time": q.get("retrieval_time", 0),
            "generation_time": q.get("generation_time", 0),
            "docs_retrieved": q.get("docs_retrieved", 0),
            "precision": round(precision_at_k(s, k, threshold), 4) if s else None,
            "recall": round(recall_at_k(s, k, threshold), 4) if s else None,
            "mrr": round(mrr(s, threshold), 4) if s else None,
            "ndcg": round(ndcg_at_k(s, k), 4) if s else None,
            "hit_rate": round(hit_rate_at_k(s, k, threshold), 4) if s else None,
        })
    return rows
