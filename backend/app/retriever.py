"""Retriever construction.

All retrieval wiring lives here (embeddings, Qdrant vector store, retriever
strategy). The RAG pipeline can then focus on orchestration and metrics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

from .embeddings import get_embeddings, get_sparse_embeddings
from .config import (
    GROQ_API_KEY,
    HYBRID_DENSE_WEIGHT,
    HYBRID_SPARSE_WEIGHT,
    LLM_MODEL,
    NUM_FUSION_QUERIES,
    QDRANT_COLLECTION,
    RETRIEVAL_K,
    USE_HYBRID_SEARCH,
)
from .qdrant_conn import get_qdrant_client


logger = logging.getLogger(__name__)

# Default to quieter logs; let the application configure logging if needed.
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.WARNING)


def build_retriever() -> Tuple[MultiQueryRetriever, QdrantVectorStore, QdrantClient]:
    """Build and validate the retriever stack.

    Returns a tuple of (retriever, vectorstore, client). Raises a ValueError if
    the API key is missing or the expected Qdrant collection does not exist.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY belum di-set. Isi di .env (lihat .env.example) sebelum menjalankan query."
        )

    # Embedding models (cached)
    embeddings = get_embeddings()
    sparse_embeddings = get_sparse_embeddings()

    # Shared Qdrant client (avoids local storage lock issues)
    client = get_qdrant_client()

    # Guard: ensure the collection exists before serving queries.
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        raise ValueError(
            "Belum ada dokumen yang diindex. Upload dan index PDF terlebih dahulu."
        )

    # Vector store wrapper with Native Hybrid Search support
    if USE_HYBRID_SEARCH:
        logger.info("Initializing QdrantVectorStore in HYBRID mode (Dense + Sparse/BM42)")
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            vector_name="",
            sparse_vector_name="sparse",
            retrieval_mode=RetrievalMode.HYBRID,
        )
    else:
        logger.info("Initializing QdrantVectorStore in DENSE mode")
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION,
            embedding=embeddings,
            vector_name="",
        )

    # Base retriever with MMR for diversity
    # Note: search_kwargs for hybrid might differ depending on langchain-qdrant version
    # but usually they are passed through to the search method.
    search_kwargs: Dict[str, Any] = {
        "k": RETRIEVAL_K,
    }
    
    base_retriever = vectorstore.as_retriever(
        search_type="mmr" if not USE_HYBRID_SEARCH else "similarity", # MMR might not be fully compatible with hybrid yet in all versions
        search_kwargs=search_kwargs
    )

    # Groq LLM
    llm = ChatGroq(
        model=LLM_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0
    )

    # MultiQuery Retriever
    prompt = PromptTemplate(
        input_variables=["question"],
        template=f"You are an AI language model assistant. Your task is to generate {NUM_FUSION_QUERIES} different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search. Provide these alternative questions separated by newlines. Original question: {{question}}"
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        prompt=prompt
    )

    logger.debug("Retriever ready (collection=%s, k=%s, mode=%s)", 
                 QDRANT_COLLECTION, RETRIEVAL_K, "hybrid" if USE_HYBRID_SEARCH else "dense")
    return retriever, vectorstore, client


def retrieve_documents(
    query: str,
    *,
    dense_retriever: MultiQueryRetriever, # Keeping name for compatibility, but it handles hybrid
    client: QdrantClient,
) -> Tuple[List[Document], Dict[str, Any]]:
    """Retrieve documents using the native hybrid strategy (if enabled)."""

    docs = dense_retriever.invoke(query)
    
    metadata: Dict[str, Any] = {
        "mode": "hybrid-native" if USE_HYBRID_SEARCH else "dense",
        "count": len(docs),
    }

    return docs, metadata


async def retrieve_documents_async(
    query: str,
    dense_retriever: MultiQueryRetriever,
    client: QdrantClient,
) -> Tuple[List[Document], Dict[str, Any]]:
    """Retrieve documents asynchronously using the native hybrid strategy."""
    
    docs = await dense_retriever.ainvoke(query)
    
    metadata: Dict[str, Any] = {
        "mode": "hybrid-native" if USE_HYBRID_SEARCH else "dense",
        "count": len(docs),
    }

    return docs, metadata