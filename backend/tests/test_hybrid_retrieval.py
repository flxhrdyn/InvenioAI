import pytest
from app.retriever import retrieve_documents, build_retriever
from app.qdrant_conn import get_qdrant_client

def test_hybrid_retrieval_returns_results():
    client = get_qdrant_client()
    # Build retriever (this will now use hybrid internally if configured)
    retriever, vectorstore, _ = build_retriever()
    
    query = "test"
    docs, meta = retrieve_documents(query, dense_retriever=retriever, client=client)
    
    assert len(docs) > 0
    # The meta["mode"] will be updated to something like "hybrid-native" in our implementation
    assert "mode" in meta
    # We will update retriever.py to set mode to 'hybrid-native'
    # For now it might say 'hybrid' or 'dense' depending on implementation
