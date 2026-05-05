"""Unit tests for hybrid rank fusion helpers."""

from langchain_core.documents import Document

from app.retriever import reciprocal_rank_fusion


def _doc(doc_id: str) -> Document:
    return Document(page_content=f"content-{doc_id}", metadata={"source": f"{doc_id}.pdf", "chunk_id": doc_id})


def test_rrf_prioritizes_consensus_docs():
    """Docs appearing in both ranked lists should rank higher."""
    a = _doc("a")
    b = _doc("b")
    c = _doc("c")
    d = _doc("d")

    fused = reciprocal_rank_fusion(
        [[a, b, c], [b, d, a]],
        rrf_k=60,
        weights=[1.0, 1.0],
        max_results=4,
    )

    # b and a appear in both lists and should be at the top.
    top_ids = [doc.metadata["chunk_id"] for doc in fused[:2]]
    assert set(top_ids) == {"a", "b"}


def test_rrf_respects_max_results():
    """Fusion result length should not exceed max_results."""
    docs_1 = [_doc(str(i)) for i in range(6)]
    docs_2 = [_doc(str(i)) for i in range(3, 9)]

    fused = reciprocal_rank_fusion(
        [docs_1, docs_2],
        rrf_k=60,
        weights=[1.0, 1.0],
        max_results=5,
    )

    assert len(fused) == 5
