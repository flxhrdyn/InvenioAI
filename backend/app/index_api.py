"""Indexing and document-management routes.

Endpoints here cover PDF upload/indexing, listing indexed sources, and clearing
the vector store.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
import uuid
from typing import Any, Dict, Literal, Optional
from qdrant_client.http import models

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from .config import (
    DELETE_UPLOADED_PDFS,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    QDRANT_URL,
    UPLOAD_DIR,
)
from .index_data import index_documents
from .qdrant_conn import close_qdrant_client, get_qdrant_client

router = APIRouter()

logger = logging.getLogger(__name__)

os.makedirs(UPLOAD_DIR, exist_ok=True)

UploadJobState = Literal["pending", "running", "succeeded", "failed"]
_upload_jobs_lock = threading.Lock()
_upload_jobs: Dict[str, Dict[str, Any]] = {}


def _set_upload_job(job: Dict[str, Any]) -> None:
    with _upload_jobs_lock:
        _upload_jobs[job["job_id"]] = job


def _get_upload_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _upload_jobs_lock:
        return _upload_jobs.get(job_id)


def _save_uploaded_pdf(file: UploadFile) -> str:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    safe_name = os.path.basename(file.filename)
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")

    file_path = os.path.join(UPLOAD_DIR, safe_name)
    if os.path.exists(file_path):
        stem, ext = os.path.splitext(safe_name)
        file_path = os.path.join(UPLOAD_DIR, f"{stem}_{uuid.uuid4().hex[:8]}{ext}")

    with open(file_path, "wb") as f:
        # Sync read keeps this endpoint compatible with normal threadpooled handling.
        f.write(file.file.read())

    return file_path


def _index_uploaded_pdf(file_path: str) -> Dict[str, str]:
    # Index into Qdrant.
    try:
        index_documents(file_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Indexing failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {type(exc).__name__}: {exc}")

    # Optional: delete local PDF after indexing (useful for deployments where
    # disk is ephemeral or not shared across instances).
    if DELETE_UPLOADED_PDFS:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        except Exception:
            # Best-effort cleanup; indexing already succeeded.
            pass

    return {"status": "PDF indexed successfully", "filename": os.path.basename(file_path)}


def _run_upload_job(job_id: str, file_path: str) -> None:
    job = _get_upload_job(job_id)
    if not job:
        return

    now = time.time()
    job["status"] = "running"
    job["updated_at"] = now
    _set_upload_job(job)

    try:
        result = _index_uploaded_pdf(file_path)
        now = time.time()
        job["status"] = "succeeded"
        job["result"] = result
        job["updated_at"] = now
        _set_upload_job(job)
    except HTTPException as exc:
        now = time.time()
        job["status"] = "failed"
        job["error"] = f"HTTP {exc.status_code}: {exc.detail}"
        job["updated_at"] = now
        _set_upload_job(job)
    except Exception as exc:
        logger.exception("Background upload job failed (job_id=%s)", job_id)
        now = time.time()
        job["status"] = "failed"
        job["error"] = f"{type(exc).__name__}: {exc}"
        job["updated_at"] = now
        _set_upload_job(job)


@router.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and index it into Qdrant.

    The file is written to `UPLOAD_DIR` first, then passed to the indexing
    pipeline. If `DELETE_UPLOADED_PDFS=1`, the local file is removed after a
    successful index.
    """

    file_path = _save_uploaded_pdf(file)
    return _index_uploaded_pdf(file_path)


@router.post("/upload/jobs")
def create_upload_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a PDF and index it in the background.

    Returns a job object immediately so clients can poll status via
    `/upload/jobs/{job_id}`.
    """

    file_path = _save_uploaded_pdf(file)

    job_id = str(uuid.uuid4())
    now = time.time()
    job: Dict[str, Any] = {
        "job_id": job_id,
        "status": "pending",
        "filename": os.path.basename(file_path),
        "created_at": now,
        "updated_at": now,
        "result": None,
        "error": None,
    }
    _set_upload_job(job)
    background_tasks.add_task(_run_upload_job, job_id, file_path)
    return job


@router.get("/upload/jobs/{job_id}")
def get_upload_job(job_id: str):
    """Return upload/indexing job state."""

    job = _get_upload_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/documents")
def list_documents():
    """List indexed document names.

    This derives the list from Qdrant payload metadata (document `source`) so
    the UI can work even when uploaded PDFs are deleted after indexing.
    """

    client = get_qdrant_client()
    try:
        existing = [c.name for c in client.get_collections().collections]
    except Exception as exc:
        logger.exception("Failed to list Qdrant collections")
        raise HTTPException(status_code=500, detail=f"Failed to query Qdrant: {type(exc).__name__}: {exc}")
    if QDRANT_COLLECTION not in existing:
        return {"documents": [], "count": 0}

    documents: set[str] = set()

    next_offset = None
    max_points = 5000
    seen_points = 0

    while True:
        try:
            points, next_offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=256,
                with_payload=True,
                with_vectors=False,
                offset=next_offset,
            )
        except Exception as exc:
            logger.exception("Failed to scroll Qdrant collection")
            raise HTTPException(status_code=500, detail=f"Failed to query Qdrant: {type(exc).__name__}: {exc}")

        if not points:
            break

        for p in points:
            payload = getattr(p, "payload", None) or {}
            if not isinstance(payload, dict):
                continue

            meta = payload.get("metadata")
            source = None
            if isinstance(meta, dict):
                # Prioritize our custom source_file basename
                source = meta.get("source_file") or meta.get("source")
            if not source:
                source = payload.get("source_file") or payload.get("source")

            if isinstance(source, str) and source:
                documents.add(os.path.basename(source))

        seen_points += len(points)
        if next_offset is None:
            break
        if seen_points >= max_points:
            break

    return {"documents": sorted(documents), "count": len(documents)}



@router.delete("/documents/delete")
def delete_document(filename: str):
    """Delete a specific document by its filename from the vector store."""
    safe_name = os.path.basename(filename)
    client = get_qdrant_client()
    try:
        # Ensure payload index exists for metadata.source_file to allow filtering for deletion
        try:
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name="metadata.source_file",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

        # Qdrant delete call using a filter on metadata.source_file
        # Qdrant delete call using a filter. 
        # We check both source_file and source to be safe with older entries.
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="metadata.source_file",
                            match=models.MatchValue(value=safe_name),
                        ),
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchText(text=safe_name),
                        ),
                    ]
                )
            ),
        )
        
        # Also try to delete from local UPLOAD_DIR if it exists
        file_path = os.path.join(UPLOAD_DIR, safe_name)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning("Could not delete physical file %s: %s", file_path, e)
            
    except Exception as exc:
        logger.exception("Failed to delete document %s", safe_name)
        raise HTTPException(
            status_code=500,
            detail=f"Delete failed: {str(exc)}",
        )
    
    return {"status": f"Document '{safe_name}' deleted successfully"}


@router.delete("/documents")
def clear_documents():
    """Delete all indexed documents.

    - In Qdrant server/cloud mode: deletes the collection.
    - In local mode: removes the on-disk storage directory.
    """

    if os.path.exists(UPLOAD_DIR):
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    try:
        if QDRANT_URL:
            # Server mode: delete collection without closing shared client.
            client = get_qdrant_client()
            existing = [c.name for c in client.get_collections().collections]
            if QDRANT_COLLECTION in existing:
                client.delete_collection(collection_name=QDRANT_COLLECTION)
        else:
            # Local mode: remove the storage directory.
            # Close open client first to avoid Windows file lock issues.
            close_qdrant_client()
            if os.path.exists(QDRANT_PATH):
                shutil.rmtree(QDRANT_PATH)
    except Exception as exc:
        logger.exception("Failed to clear documents")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear vector store: {type(exc).__name__}: {exc}",
        )

    return {"status": "Documents and vector store cleared"}