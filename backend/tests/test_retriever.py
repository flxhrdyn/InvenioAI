import pytest
from langchain_core.documents import Document
from app.retriever import reciprocal_rank_fusion

def test_rrf_higher_ranked_docs_get_higher_scores():
    # Setup
    list1 = [Document(page_content="doc A"), Document(page_content="doc B")]
    list2 = [Document(page_content="doc A"), Document(page_content="doc C")]
    
    # Execution
    # weight defaults to 1.0. doc A is rank 1 in both.
    result = reciprocal_rank_fusion([list1, list2], rrf_k=60, weights=[1.0, 1.0], max_results=10)
    
    # Assertions
    assert len(result) == 3
    # doc A should be first because it is top ranked in both lists
    assert result[0].page_content == "doc A"

def test_rrf_unique_docs_have_lower_scores_than_overlapping_ones():
    # Setup
    list1 = [Document(page_content="doc A"), Document(page_content="doc B")]
    list2 = [Document(page_content="doc A"), Document(page_content="doc C")]
    
    # Execution
    result = reciprocal_rank_fusion([list1, list2], rrf_k=60, weights=[1.0, 1.0], max_results=10)
    
    # Assertions
    # doc B and doc C are unique, so their scores should be lower than doc A
    # Their order doesn't strictly matter as much as being after doc A
    assert result[0].page_content == "doc A"
    assert result[1].page_content in ["doc B", "doc C"]
    assert result[2].page_content in ["doc B", "doc C"]

def test_rrf_max_results_limit():
    # Setup
    list1 = [Document(page_content=f"doc {i}") for i in range(10)]
    list2 = [Document(page_content=f"doc {i+10}") for i in range(10)]
    
    # Execution
    result = reciprocal_rank_fusion([list1, list2], rrf_k=60, weights=[1.0, 1.0], max_results=5)
    
    # Assertions
    assert len(result) == 5

def test_rrf_handles_empty_input():
    result = reciprocal_rank_fusion([], rrf_k=60, weights=[1.0, 1.0], max_results=5)
    assert result == []

    result2 = reciprocal_rank_fusion([[], []], rrf_k=60, weights=[1.0, 1.0], max_results=5)
    assert result2 == []
