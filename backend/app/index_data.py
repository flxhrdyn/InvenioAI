"""PDF indexing pipeline.

Given a local PDF file path, it loads the pages, splits them into overlapping
chunks, embeds the chunks, and upserts them into the configured Qdrant
collection.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language
)
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models
from langchain_core.documents import Document

from .embeddings import get_embeddings, get_sparse_embeddings
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
    """Load and chunk PDF documents using OpenDataLoader (Markdown) and Structure-Aware Splitting."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # Use OpenDataLoader for high-fidelity Markdown extraction
    loader = OpenDataLoaderPDFLoader(
        file_path=file_path,
        format="markdown"
    )
    docs = loader.load()
    
    # Structure-Aware Splitting (Step 1: Headers)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    final_chunks = []
    
    # Recursive Splitter for Markdown (Step 2: Size)
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )
    
    file_name = path.name
    for doc in docs:
        # Split by headers
        header_splits = markdown_splitter.split_text(doc.page_content)
        
        for split in header_splits:
            # Further split by size while respecting Markdown structure
            sub_chunks = text_splitter.create_documents(
                [split.page_content], 
                metadatas=[split.metadata]
            )
            
            for chunk in sub_chunks:
                # Merge page/source metadata from original doc
                chunk.metadata.update(doc.metadata)
                chunk.metadata["source_file"] = file_name
                
                # Normalize page number (OpenDataLoader usually adds 'page' to metadata)
                if "page" in chunk.metadata:
                    chunk.metadata["page_number"] = chunk.metadata["page"] + 1
                    
                final_chunks.append(chunk)
            
    return final_chunks


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
                    vectors_config={
                        "dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams()
                    },
                )

                logger.info("Qdrant collection created with Hybrid Search support (Dense + Sparse)")
            else:
                logger.info("Using existing Qdrant collection: %s", QDRANT_COLLECTION)

            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=QDRANT_COLLECTION,
                embedding=embeddings,
                sparse_embedding=get_sparse_embeddings(),
                vector_name="dense",
                sparse_vector_name="sparse",
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

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        logging.basicConfig(level=logging.INFO)
        index_documents(sys.argv[1])
    else:
        print("Usage: python backend/app/index_data.py <pdf_path>")