import logging
import time
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from qdrant_client import models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams

from .config import (
    INDEXING_BATCH_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    QDRANT_COLLECTION,
)
from .embeddings import get_embeddings, get_sparse_embeddings
from .qdrant_conn import get_qdrant_client

from typing import Callable, Optional

logger = logging.getLogger(__name__)


def process_pdf_documents(
    file_path: str, 
    status_callback: Optional[Callable[[str], None]] = None
) -> list[Document]:
    """Load and chunk PDF documents using LlamaParse (Markdown) and Structure-Aware Splitting."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    if status_callback:
        status_callback("parsing")
        
    # Use LlamaParse with high-fidelity settings
    from llama_parse import LlamaParse
    
    # 1. Initialize Parser
    parser = LlamaParse(
        result_type="markdown", 
        num_workers=4,
        verbose=True,
        language="en",
        user_prompt="Please extract the content of this document with high fidelity. Pay special attention to tables and ensure they are formatted correctly in Markdown. Preserve the logical structure and headers of the document. DO NOT include any page headers, page footers, or page numbers in the extracted text. Exclude repeating document titles or copyright notices at the top or bottom of pages."
    )
    
    # 2. Load Documents (Cloud Parsing)
    llama_docs = parser.load_data(file_path)
    
    all_header_splits = []
    
    # 3. Structure-Aware Splitting (Step 1: Headers)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    for i, doc in enumerate(llama_docs):
        # Ambil nomor halaman dari metadata LlamaIndex (dimulai dari 1)
        page_num = doc.metadata.get("page_number", str(i + 1))
        
        # Split per halaman agar metadata 'page_label' tidak tercampur
        page_splits = header_splitter.split_text(doc.text)
        
        for split in page_splits:
            split.metadata.update({
                "source": file_path,
                "source_file": path.name,  # Essential for the /documents API
                "page_label": page_num,
                "file_name": path.name
            })
            all_header_splits.append(split)
    
    # 4. Semantic/Length-based Splitting (Step 2)
    # Using header_splits as input to ensure headers are preserved in chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    
    final_chunks = text_splitter.split_documents(all_header_splits)
    
    # Final metadata normalization (Ensure 'Header 1' etc are consistent)
    for chunk in final_chunks:
        # Ensure we don't have None values in metadata which can break Qdrant
        for k, v in list(chunk.metadata.items()):
            if v is None:
                chunk.metadata[k] = ""
                
    return final_chunks


def index_documents(
    file_path: str, 
    status_callback: Optional[Callable[[str], None]] = None
) -> None:
    """Index a single PDF into Qdrant using the semantic pipeline."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    total_start = time.monotonic()
    logger.info("Indexing PDF: %s", path.name)

    # Process (Load + Semantic Chunk)
    chunks = process_pdf_documents(file_path, status_callback=status_callback)
    logger.info("Created %d semantic chunks", len(chunks))

    if status_callback:
        status_callback("indexing")

    # Initialize Qdrant Collection
    client = get_qdrant_client()
    embeddings = get_embeddings()
    sparse_embeddings_model = get_sparse_embeddings()

    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        logger.info("Creating collection: %s", QDRANT_COLLECTION)
        # Dummy vector to get dimension
        dim = len(embeddings.embed_query("test"))
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            sparse_vectors_config={
                "sparse": SparseVectorParams()
            }
        )

    # Batch Indexing
    for i in range(0, len(chunks), INDEXING_BATCH_SIZE):
        batch = chunks[i : i + INDEXING_BATCH_SIZE]
        texts = [c.page_content for c in batch]
        metadatas = [c.metadata for c in batch]

        # Generate Dense Embeddings
        vectors = embeddings.embed_documents(texts)
        
        # Generate Sparse Embeddings
        sparse_vectors = sparse_embeddings_model.embed_documents(texts)

        points = []
        for j, (text, meta, vector, sparse_vector) in enumerate(zip(texts, metadatas, vectors, sparse_vectors)):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "": vector, # Unnamed dense vector
                        "sparse": models.SparseVector(
                            indices=sparse_vector.indices, 
                            values=sparse_vector.values
                        )
                    },
                    payload={
                        "page_content": text,
                        "metadata": meta,
                    },
                )
            )

        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points
        )
        logger.info("Indexed batch %d/%d", i + len(batch), len(chunks))

    total_time = time.monotonic() - total_start
    logger.info("Indexing complete for %s (Time: %.2fs)", path.name, total_time)


def delete_all_documents() -> None:
    """Wipe the current collection in Qdrant."""
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in collections:
        logger.info("Deleting collection: %s", QDRANT_COLLECTION)
        client.delete_collection(QDRANT_COLLECTION)
    else:
        logger.warning("Collection %s not found, nothing to delete", QDRANT_COLLECTION)