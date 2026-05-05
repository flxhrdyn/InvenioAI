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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .embeddings import get_embeddings
from .index_api import router as index_router
from .config import PRELOAD_EMBEDDINGS_ON_STARTUP
from .rag_pipeline import rag_pipeline
from .qdrant_conn import close_qdrant_client
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

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    if not PRELOAD_EMBEDDINGS_ON_STARTUP:
        logger.info("Embedding preload skipped (INVENIOAI_PRELOAD_EMBEDDINGS=0)")
    else:
        try:
            get_embeddings()
            logger.info("Embedding model preloaded")
        except Exception:
            logger.warning("Embedding preload failed; falling back to lazy init", exc_info=True)
            
    yield
    
    # Shutdown logic
    close_qdrant_client()


app = FastAPI(title="InvenioAI API", lifespan=lifespan)
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