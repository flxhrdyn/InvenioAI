from app.embeddings import get_sparse_embeddings

def test_sparse_embedding_output():
    model = get_sparse_embeddings()
    query = "halo dunia"
    # FastEmbed SparseTextEmbedding.embed returns a generator of dicts
    # In our wrapper, we might want to return the same or adapt it.
    # The plan says: vector = list(model.embed_query(query))
    # But FastEmbed doesn't have embed_query by default, that's LangChain style.
    # If we use SparseTextEmbedding directly, it's .embed()
    
    # Let's see what the plan expects:
    # vector = list(model.embed_query(query))
    # assert isinstance(vector[0], dict)
    
    vector = model.embed_query(query)
    assert hasattr(vector, "indices")
    assert hasattr(vector, "values")
    assert len(vector.indices) > 0
