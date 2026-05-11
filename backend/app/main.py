"""FastAPI application entrypoint.

Wires the API router and exposes two ways to query the RAG pipeline:

- `POST /query` for a simple request/response flow.
- `POST /query/jobs` for background execution with polling.

Job state is stored in-memory, so it resets on process restart.
"""

import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Literal, Optional
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .embeddings import get_embeddings, get_sparse_embeddings
from .index_api import router as index_router
from .config import PRELOAD_EMBEDDINGS_ON_STARTUP
from .rag_pipeline import rag_pipeline
from .qdrant_conn import close_qdrant_client, get_qdrant_client
from .reranker import preload_reranker
from .metrics import (
    load_metrics,
    get_avg_response_time,
    get_avg_retrieval_time,
    get_avg_generation_time,
    get_avg_docs_retrieved,
    get_avg_chunks_processed,
    get_retrieval_efficiency,
    get_generation_efficiency,
    compute_ir_metrics
)

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(message)s"
)
logger = logging.getLogger(__name__)


def preload_all_models() -> None:
    """Preload all model singletons (Dense, Sparse, Reranker) and Qdrant connection."""
    logger.info("Starting full model preload...")
    get_embeddings()
    get_sparse_embeddings()
    preload_reranker()
    get_qdrant_client()
    logger.info("All models and database connections preloaded successfully.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info(f"Lifespan starting. Preload setting: {PRELOAD_EMBEDDINGS_ON_STARTUP}")
    
    if not PRELOAD_EMBEDDINGS_ON_STARTUP:
        logger.info("Embedding preload skipped (INVENIOAI_PRELOAD_EMBEDDINGS=0)")
    else:
        try:
            preload_all_models()
        except Exception:
            logger.warning("Full model preload failed; falling back to lazy init", exc_info=True)
    
    # Reconcile metrics: Sync total_documents_indexed from Qdrant persistent storage
    try:
        from .qdrant_conn import get_qdrant_client
        from .metrics import sync_indexed_docs_count
        from .config import QDRANT_COLLECTION
        
        client = get_qdrant_client()
        offset = None
        unique_files = set()
        logger.info(f"Starting reconciliation for collection: {QDRANT_COLLECTION}")
        
        while True:
            points, offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            for p in points:
                if p.payload:
                    # Check common keys for filename
                    # Note: index_data.py uses 'source_file'
                    fname = (
                        p.payload.get("source_file") or 
                        p.payload.get("file") or 
                        p.payload.get("filename") or 
                        p.payload.get("source") or
                        p.payload.get("metadata", {}).get("source_file") or
                        p.payload.get("metadata", {}).get("file")
                    )
                    if fname:
                        unique_files.add(fname)
            if offset is None:
                break
        
        count = len(unique_files)
        logger.info(f"Reconciliation successful: Found {count} unique documents ({list(unique_files)[:5]}...). Syncing metrics.")
        sync_indexed_docs_count(count)
    except Exception as e:
        logger.error(f"Critical failure during metrics reconciliation: {e}", exc_info=True)
            
    yield
    
    # Shutdown logic
    close_qdrant_client()


app = FastAPI(title="InvenioAI API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(index_router)




class Query(BaseModel):
    question: str
    history: List[str] = Field(default_factory=list)


@app.post("/query")
def query(q: Query) -> Dict[str, Any]:
    try:
        result = rag_pipeline(q.question, q.history)
        return {
            "answer": result["answer"],
            "sources": result.get("sources", ""),
            "metrics": result.get("metrics", {}),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("/query failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/query/stream", tags=["query"])
async def query_stream_endpoint(request: Query):
    """Execute a query and stream the response using Server-Sent Events (SSE)."""
    
    # We use a generator that returns the async pipeline stream
    async def event_generator():
        from .rag_pipeline import rag_pipeline_stream_async
        async for chunk in rag_pipeline_stream_async(request.question, request.history):
            # The pipeline yields JSON strings with a trailing newline
            # We format it as SSE "data: <json>\n\n"
            json_str = chunk.strip()
            yield f"data: {json_str}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@app.post("/metrics/sync", tags=["analytics"])
async def sync_metrics_endpoint():
    """Manually trigger a sync of indexed document counts from Qdrant."""
    try:
        from .qdrant_conn import get_qdrant_client
        from .metrics import sync_indexed_docs_count
        from .config import QDRANT_COLLECTION
        
        client = get_qdrant_client()
        offset = None
        unique_files = set()
        
        while True:
            points, offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            for p in points:
                if p.payload:
                    fname = (
                        p.payload.get("source_file") or 
                        p.payload.get("file") or 
                        p.payload.get("filename") or 
                        p.payload.get("source") or
                        p.payload.get("metadata", {}).get("source_file") or
                        p.payload.get("metadata", {}).get("file")
                    )
                    if fname:
                        unique_files.add(fname)
            if offset is None:
                break
        
        count = len(unique_files)
        sync_indexed_docs_count(count)
        return {"status": "success", "count": count}
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/metrics", tags=["analytics"])
def get_metrics() -> Dict[str, Any]:
    """Return aggregate RAG performance and quality metrics."""
    metrics = load_metrics()
    
    # Compute aggregate IR metrics
    ir_metrics = compute_ir_metrics(metrics.get("query_history", []))
    
    return {
        "total_queries": metrics.get("total_queries", 0),
        "total_documents_indexed": metrics.get("total_documents_indexed", 0),
        "avg_response_time": get_avg_response_time(),
        "avg_retrieval_time": get_avg_retrieval_time(),
        "avg_generation_time": get_avg_generation_time(),
        "avg_docs_retrieved": get_avg_docs_retrieved(),
        "avg_chunks_processed": get_avg_chunks_processed(),
        "retrieval_efficiency_pct": get_retrieval_efficiency(),
        "generation_efficiency_pct": get_generation_efficiency(),
        "ir_quality": ir_metrics,
        "query_history": metrics.get("query_history", [])[-10:]  # Last 10 for quick view
    }


@app.get("/")
def root():
    return {"status": "InvenioAI API running"}