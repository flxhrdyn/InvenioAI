"""PDF indexing pipeline.

Given a local PDF file path, it loads the pages, splits them into overlapping
chunks, embeds the chunks, and upserts them into the configured Qdrant
collection.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams
from langchain_core.documents import Document

from .embeddings import get_embeddings
from .config import QDRANT_COLLECTION, INDEXING_BATCH_SIZE, CHUNK_SIZE, CHUNK_OVERLAP
from .metrics import log_document_indexed
from .qdrant_conn import get_qdrant_client, is_qdrant_client_closed_error, recreate_qdrant_client


logger = logging.getLogger(__name__)


def _collection_exists(client, collection_name: str) -> bool:
    if hasattr(client, "collection_exists"):
        return bool(client.collection_exists(collection_name=collection_name))

    collections = client.get_collections().collections
    return collection_name in [collection.name for collection in collections]


def process_pdf_documents(file_path: str) -> list[Document]:
    """Load and chunk PDF documents using Recursive Character Splitter."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    
    # Recursive Character Splitting (Fast & Structured)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        add_start_index=True
    )
    
    chunked_docs = text_splitter.split_documents(docs)
    
    # Metadata Injection (Page Tracking)
    file_name = path.name
    for doc in chunked_docs:
        doc.metadata["source_file"] = file_name
        # PyMuPDFLoader already adds 'page' to metadata (0-indexed)
        # We ensure it's there and maybe make it 1-indexed for the UI later
        if "page" in doc.metadata:
            doc.metadata["page_number"] = doc.metadata["page"] + 1
            
    return chunked_docs


def index_documents(file_path: str) -> None:
    """Index a single PDF into Qdrant using the semantic pipeline."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    total_start = time.monotonic()
    logger.info("Indexing PDF: %s", path.name)

    # Process (Load + Semantic Chunk)
    chunks = process_pdf_documents(file_path)
    logger.info("Created %d semantic chunks", len(chunks))

    # Embedding model (cached)
    embeddings = get_embeddings()

    max_attempts = 2
    for attempt in range(max_attempts):
        client = get_qdrant_client() if attempt == 0 else recreate_qdrant_client()

        try:
            if not _collection_exists(client, QDRANT_COLLECTION):
                logger.info("Creating Qdrant collection: %s", QDRANT_COLLECTION)

                # Determine embedding vector size
                sample_vector = embeddings.embed_query("sample")
                vector_size = len(sample_vector)

                client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )

                logger.info("Qdrant collection created")
            else:
                logger.info("Using existing Qdrant collection: %s", QDRANT_COLLECTION)

            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=QDRANT_COLLECTION,
                embedding=embeddings,
            )
            
            upsert_start = time.monotonic()
            vectorstore.add_documents(chunks, batch_size=INDEXING_BATCH_SIZE)
            logger.info(
                "Indexed %d chunks into Qdrant (batch=%d, %.2fs)",
                len(chunks),
                INDEXING_BATCH_SIZE,
                time.monotonic() - upsert_start,
            )
            break
        except Exception as exc:
            if attempt < (max_attempts - 1) and is_qdrant_client_closed_error(exc):
                logger.warning(
                    "Qdrant client was closed during indexing; retrying..."
                )
                continue
            raise

    # Update metrics
    log_document_indexed()
    logger.info("Indexing complete (%.2fs total)", time.monotonic() - total_start)