"""Unit tests for the metrics helpers.

These tests cover the file-backed counters, query history logging, and the
derived averages/efficiency helpers exposed by :mod:`app.metrics`.
"""
import os
import pytest
from app.metrics import (
    load_metrics,
    save_metrics,
    log_query,
    log_document_indexed,
    get_avg_response_time,
    get_avg_retrieval_time,
    get_avg_generation_time,
    get_avg_docs_retrieved,
    get_avg_chunks_processed,
    get_retrieval_efficiency,
    get_generation_efficiency,
    reset_metrics,
    METRICS_FILE
)


@pytest.fixture
def clean_metrics():
    """Ensure the metrics file is removed before and after a test."""
    # Clean up before test.
    if os.path.exists(METRICS_FILE):
        os.remove(METRICS_FILE)
    
    yield
    
    # Clean up after test.
    if os.path.exists(METRICS_FILE):
        os.remove(METRICS_FILE)


def test_load_metrics_empty(clean_metrics):
    """Load defaults when the metrics file does not exist."""
    metrics = load_metrics()
    
    assert metrics["total_queries"] == 0
    assert metrics["total_documents_indexed"] == 0
    assert metrics["total_response_time"] == 0
    assert metrics["total_retrieval_time"] == 0
    assert metrics["total_generation_time"] == 0
    assert metrics["total_docs_retrieved"] == 0
    assert metrics["total_chunks_processed"] == 0
    assert isinstance(metrics["query_history"], list)
    assert len(metrics["query_history"]) == 0


def test_save_and_load_metrics(clean_metrics):
    """Persist metrics to disk and load them back."""
    test_metrics = {
        "total_queries": 5,
        "total_documents_indexed": 2,
        "total_response_time": 10.5,
        "total_retrieval_time": 3.0,
        "total_generation_time": 5.5,
        "total_docs_retrieved": 25,
        "total_chunks_processed": 15,
        "query_history": []
    }
    
    save_metrics(test_metrics)
    loaded = load_metrics()
    
    assert loaded["total_queries"] == 5
    assert loaded["total_documents_indexed"] == 2
    assert loaded["total_response_time"] == 10.5
    assert loaded["total_retrieval_time"] == 3.0
    assert loaded["total_generation_time"] == 5.5
    assert loaded["total_docs_retrieved"] == 25
    assert loaded["total_chunks_processed"] == 15


def test_log_query(clean_metrics):
    """Log a single query and update the aggregate totals."""
    log_query("What is Python?", 2.5, 150, 1.0, 1.2, 5, 3)
    
    metrics = load_metrics()
    assert metrics["total_queries"] == 1
    assert metrics["total_response_time"] == 2.5
    assert metrics["total_retrieval_time"] == 1.0
    assert metrics["total_generation_time"] == 1.2
    assert metrics["total_docs_retrieved"] == 5
    assert metrics["total_chunks_processed"] == 3
    assert len(metrics["query_history"]) == 1
    
    query = metrics["query_history"][0]
    assert "What is Python?" in query["question"]
    assert query["response_time"] == 2.5
    assert query["retrieval_time"] == 1.0
    assert query["generation_time"] == 1.2
    assert query["answer_length"] == 150
    assert query["docs_retrieved"] == 5
    assert query["chunks_processed"] == 3


def test_log_multiple_queries(clean_metrics):
    """Log multiple queries and keep totals consistent."""
    for i in range(5):
        log_query(f"Question {i}", 1.0 + i, 100 + i * 10, 0.5 + i * 0.1, 0.4 + i * 0.1, 3 + i, 2 + i)
    
    metrics = load_metrics()
    assert metrics["total_queries"] == 5
    assert metrics["total_response_time"] == 15.0  # 1+2+3+4+5
    assert len(metrics["query_history"]) == 5


def test_log_query_truncates_long_question(clean_metrics):
    """Truncate overly long questions before storing in history."""
    long_question = "A" * 200
    log_query(long_question, 1.5, 100, 0.5, 0.8, 4, 2)
    
    metrics = load_metrics()
    query = metrics["query_history"][0]
    assert len(query["question"]) == 100  # Should be truncated


def test_log_query_max_history(clean_metrics):
    """Test that query history is limited to 100 entries"""
    # Log 105 queries
    for i in range(105):
        log_query(f"Query {i}", 1.0, 100, 0.5, 0.4, 3, 2)
    
    metrics = load_metrics()
    assert len(metrics["query_history"]) == 100
    # Should keep the last 100
    assert "Query 5" in metrics["query_history"][0]["question"]


def test_log_document_indexed(clean_metrics):
    """Test logging document indexing"""
    log_document_indexed()
    
    metrics = load_metrics()
    assert metrics["total_documents_indexed"] == 1
    
    log_document_indexed()
    log_document_indexed()
    
    metrics = load_metrics()
    assert metrics["total_documents_indexed"] == 3


def test_get_avg_response_time_no_queries(clean_metrics):
    """Test average response time with no queries"""
    avg = get_avg_response_time()
    assert avg == 0.0


def test_get_avg_response_time_with_queries(clean_metrics):
    """Test average response time calculation"""
    log_query("Q1", 2.0, 100, 0.8, 1.0, 5, 3)
    log_query("Q2", 4.0, 100, 1.5, 2.0, 5, 3)
    log_query("Q3", 3.0, 100, 1.2, 1.5, 5, 3)
    
    avg = get_avg_response_time()
    assert avg == 3.0  # (2+4+3)/3


def test_get_avg_retrieval_time(clean_metrics):
    """Test average retrieval time calculation"""
    log_query("Q1", 3.0, 100, 1.0, 1.8, 5, 3)
    log_query("Q2", 4.0, 100, 1.5, 2.0, 6, 4)
    log_query("Q3", 3.5, 100, 1.2, 2.0, 5, 3)
    
    avg = get_avg_retrieval_time()
    assert avg == 1.23  # (1.0+1.5+1.2)/3 rounded


def test_get_avg_generation_time(clean_metrics):
    """Test average generation time calculation"""
    log_query("Q1", 3.0, 100, 1.0, 1.8, 5, 3)
    log_query("Q2", 4.0, 100, 1.5, 2.0, 6, 4)
    log_query("Q3", 3.5, 100, 1.2, 2.0, 5, 3)
    
    avg = get_avg_generation_time()
    assert avg == 1.93  # (1.8+2.0+2.0)/3 rounded


def test_get_avg_docs_retrieved(clean_metrics):
    """Test average documents retrieved calculation"""
    log_query("Q1", 3.0, 100, 1.0, 1.8, 5, 3)
    log_query("Q2", 4.0, 100, 1.5, 2.0, 7, 4)
    log_query("Q3", 3.5, 100, 1.2, 2.0, 6, 3)
    
    avg = get_avg_docs_retrieved()
    assert avg == 6.0  # (5+7+6)/3


def test_get_avg_chunks_processed(clean_metrics):
    """Test average chunks processed calculation"""
    log_query("Q1", 3.0, 100, 1.0, 1.8, 5, 3)
    log_query("Q2", 4.0, 100, 1.5, 2.0, 7, 5)
    log_query("Q3", 3.5, 100, 1.2, 2.0, 6, 4)
    
    avg = get_avg_chunks_processed()
    assert avg == 4.0  # (3+5+4)/3


def test_get_retrieval_efficiency(clean_metrics):
    """Test retrieval efficiency calculation"""
    log_query("Q1", 4.0, 100, 1.0, 2.5, 5, 3)
    log_query("Q2", 6.0, 100, 2.0, 3.0, 6, 4)
    
    efficiency = get_retrieval_efficiency()
    # (1.0 + 2.0) / (4.0 + 6.0) * 100 = 30%
    assert efficiency == 30.0


def test_get_generation_efficiency(clean_metrics):
    """Test generation efficiency calculation"""
    log_query("Q1", 4.0, 100, 1.0, 2.5, 5, 3)
    log_query("Q2", 6.0, 100, 2.0, 3.0, 6, 4)
    
    efficiency = get_generation_efficiency()
    # (2.5 + 3.0) / (4.0 + 6.0) * 100 = 55%
    assert efficiency == 55.0


def test_reset_metrics(clean_metrics):
    """Test resetting all metrics"""
    # Add some data
    log_query("Test", 1.5, 100, 0.5, 0.8, 4, 2)
    log_document_indexed()
    
    # Reset
    reset_metrics()
    
    # Verify everything is reset
    metrics = load_metrics()
    assert metrics["total_queries"] == 0
    assert metrics["total_documents_indexed"] == 0
    assert metrics["total_response_time"] == 0
    assert metrics["total_retrieval_time"] == 0
    assert metrics["total_generation_time"] == 0
    assert metrics["total_docs_retrieved"] == 0
    assert metrics["total_chunks_processed"] == 0
    assert len(metrics["query_history"]) == 0


class TestMetricsEdgeCases:
    """Test edge cases for metrics"""
    
    def test_negative_response_time(self, clean_metrics):
        """Test handling of negative response time"""
        log_query("Test", -1.0, 100, 0.5, 0.8, 4, 2)
        metrics = load_metrics()
        # Should still log, but value will be negative
        assert metrics["total_response_time"] == -1.0
    
    def test_zero_response_time(self, clean_metrics):
        """Test handling of zero response time"""
        log_query("Test", 0.0, 100, 0.0, 0.0, 0, 0)
        avg = get_avg_response_time()
        assert avg == 0.0
    
    def test_zero_efficiency_no_time(self, clean_metrics):
        """Test efficiency calculation when no time recorded"""
        efficiency = get_retrieval_efficiency()
        assert efficiency == 0.0
    
    def test_corrupted_metrics_file(self, clean_metrics):
        """Test loading corrupted metrics file"""
        # Write invalid JSON
        with open(METRICS_FILE, 'w') as f:
            f.write("invalid json {{{")
        
        # Should return default metrics
        metrics = load_metrics()
        assert metrics["total_queries"] == 0
