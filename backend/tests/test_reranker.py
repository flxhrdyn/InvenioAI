from langchain_core.documents import Document
from app.reranker import rerank
from app.config import RERANKER_MODEL

def test_reranker_model_validity():
    """Ensure the configured RERANKER_MODEL is valid and doesn't trigger fallback."""
    query = "What is the capital of France?"
    docs = [
        Document(page_content="Paris is the capital and most populous city of France.", metadata={"source": "doc1"}),
        Document(page_content="The economy of France is highly developed and free-market-oriented.", metadata={"source": "doc2"}),
        Document(page_content="Exploring the scenic routes and vineyards of rural France.", metadata={"source": "doc3"})
    ]
    
    ranked_docs, scores = rerank(query, docs)
    
    # In the broken state (invalid model), reranking fails silently,
    # falling back to original order and returning all 0.0 scores.
    # Therefore, asserting that at least one score is greater than 0.0
    # ensures the reranker is actually online and working.
    assert len(ranked_docs) > 0, "Should return ranked documents"
    assert len(scores) == len(ranked_docs), "Should have a score for each doc"
    assert any(s > 0.0 for s in scores), f"Reranker failed (fallback scores are all 0.0). Model: {RERANKER_MODEL}"
