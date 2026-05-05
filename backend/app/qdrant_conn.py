"""Qdrant client factory.

The app uses a process-wide singleton client to avoid local storage lock
problems (especially on Windows) and to keep connection setup consistent.

If `QDRANT_URL` is set we connect to a server/cloud instance; otherwise we fall
back to local on-disk storage at `QDRANT_PATH`.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from qdrant_client import QdrantClient

from .config import QDRANT_API_KEY, QDRANT_PATH, QDRANT_URL


logger = logging.getLogger(__name__)

_client_lock = threading.Lock()
_client: Optional[QdrantClient] = None


def _create_qdrant_client() -> QdrantClient:
    if QDRANT_URL:
        # Cloud/server mode.
        # Default to HTTP/REST for reliability (some networks block gRPC).
        prefer_grpc = os.getenv("QDRANT_PREFER_GRPC", "0").strip() == "1"
        return QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=prefer_grpc,
        )

    return QdrantClient(path=QDRANT_PATH)


def is_qdrant_client_closed_error(exc: BaseException) -> bool:
    """Return True if an exception indicates the Qdrant client is closed."""

    msg = str(exc).strip().lower()
    return "client has been closed" in msg or "client is closed" in msg


def get_qdrant_client() -> QdrantClient:
    """Return a process-wide Qdrant client.

    Local Qdrant storage (path=...) cannot be safely opened by multiple client
    instances concurrently. Reusing a singleton client prevents lock errors when
    multiple requests hit the API at the same time.

    If QDRANT_URL is set, uses Qdrant server mode which supports concurrent
    access from multiple processes.
    """

    global _client
    with _client_lock:
        if _client is None:
            _client = _create_qdrant_client()
        return _client


def recreate_qdrant_client() -> QdrantClient:
    """Force-close old client (if any) and create a fresh one."""

    global _client
    with _client_lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:
                logger.warning("Ignoring failure while closing stale Qdrant client", exc_info=True)

        _client = _create_qdrant_client()
        return _client


def close_qdrant_client() -> None:
    """Close and reset the cached Qdrant client (best-effort)."""
    global _client
    with _client_lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:
                logger.warning("Ignoring failure while closing Qdrant client", exc_info=True)
            finally:
                _client = None
