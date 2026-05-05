"""Application configuration.

Importing `app.config` loads `.env` and sets a few environment defaults used by
the Hugging Face stack. Those defaults need to be in place before heavy ML
libraries are imported.
"""

from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv(override=False)

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: str = "0") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, min_value: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return max(default, min_value)
    try:
        value = int(raw)
    except ValueError:
        return max(default, min_value)
    return max(value, min_value)


def _env_float(name: str, default: float, *, min_value: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return max(default, min_value)
    try:
        value = float(raw)
    except ValueError:
        return max(default, min_value)
    return max(value, min_value)


def _env_str(name: str, default: str) -> str:
    raw = (os.getenv(name) or "").strip()
    return raw if raw else default


def _configure_huggingface_hub_defaults() -> None:
    """Set safe HF Hub defaults (timeouts, offline mode, telemetry).

    Values are applied via `os.environ.setdefault(...)` so user-provided env
    vars always take precedence.
    """

    # Avoid noisy telemetry by default.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    # Hugging Face Hub HTTP timeouts. The default read timeout (10s) is often
    # too aggressive on slow networks.
    os.environ.setdefault("HF_HUB_CONNECT_TIMEOUT", os.getenv("HF_HUB_CONNECT_TIMEOUT", "30"))
    os.environ.setdefault("HF_HUB_READ_TIMEOUT", os.getenv("HF_HUB_READ_TIMEOUT", "60"))

    # Optional offline mode (if you have pre-downloaded models into the cache).
    if os.getenv("HF_HUB_OFFLINE") == "1":
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


_configure_huggingface_hub_defaults()

# Qdrant
QDRANT_PATH = str(BASE_DIR / "qdrant_storage")
QDRANT_COLLECTION = "invenioai_collection"

# Optional: use Qdrant server/cloud instead of local storage.
_qdrant_url = (os.getenv("QDRANT_URL") or "").strip()
QDRANT_URL = _qdrant_url or None  # e.g. http://localhost:6333

_qdrant_api_key = (os.getenv("QDRANT_API_KEY") or "").strip()
QDRANT_API_KEY = _qdrant_api_key or None
UPLOAD_DIR = str(BASE_DIR / "uploaded_docs")
METRICS_FILE = str(BASE_DIR / "metrics.json")

# When enabled, uploaded PDFs are deleted from local disk after indexing.
# Useful for deployments with ephemeral disks.
DELETE_UPLOADED_PDFS = (
    (os.getenv("INVENIOAI_DELETE_UPLOADED_PDFS") or os.getenv("DELETE_UPLOADED_PDFS") or "0").strip() == "1"
)

# Chunking
CHUNK_SIZE = _env_int("INVENIOAI_CHUNK_SIZE", default=800, min_value=128)
CHUNK_OVERLAP = _env_int("INVENIOAI_CHUNK_OVERLAP", default=100, min_value=0)
if CHUNK_OVERLAP >= CHUNK_SIZE:
    CHUNK_OVERLAP = max(0, CHUNK_SIZE // 4)

# Smaller default batch is safer for constrained runtimes (HF Spaces free tier,
# small containers) and often avoids long stalls from memory pressure.
INDEXING_BATCH_SIZE = _env_int("INVENIOAI_INDEXING_BATCH_SIZE", default=32, min_value=8)

# Startup preload can make the first request faster, but on constrained
# deployments it may increase cold-start time and memory pressure.
PRELOAD_EMBEDDINGS_ON_STARTUP = _env_bool("INVENIOAI_PRELOAD_EMBEDDINGS", default="0")

# Retrieval
RETRIEVAL_K = 10

# Hybrid retrieval (dense + lexical) settings.
# Hybrid uses weighted reciprocal-rank fusion (RRF) before reranking and is
# enabled by default. Set INVENIOAI_ENABLE_HYBRID_SEARCH=0 to force dense-only.
USE_HYBRID_SEARCH = _env_bool("INVENIOAI_ENABLE_HYBRID_SEARCH", default="1")
HYBRID_LEXICAL_K = _env_int("INVENIOAI_HYBRID_LEXICAL_K", default=10, min_value=1)
HYBRID_FUSION_LIMIT = _env_int("INVENIOAI_HYBRID_FUSION_LIMIT", default=20, min_value=1)
HYBRID_MAX_LEXICAL_DOCS = _env_int("INVENIOAI_HYBRID_MAX_LEXICAL_DOCS", default=3000, min_value=100)
HYBRID_RRF_K = _env_int("INVENIOAI_HYBRID_RRF_K", default=60, min_value=1)
HYBRID_DENSE_WEIGHT = _env_float("INVENIOAI_HYBRID_DENSE_WEIGHT", default=1.0, min_value=0.0)
HYBRID_LEXICAL_WEIGHT = _env_float("INVENIOAI_HYBRID_LEXICAL_WEIGHT", default=1.0, min_value=0.0)

# Reranking
RERANK_TOP_K = 5

# Models
LLM_MODEL = _env_str("INVENIOAI_LLM_MODEL", "llama3-8b-8192")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL = "ms-marco-MultiBERT-L-12"

# API Keys
_groq_api_key = (os.getenv("GROQ_API_KEY") or "").strip()
GROQ_API_KEY = _groq_api_key or None

# Caching
CACHE_TYPE = _env_str("CACHE_TYPE", "diskcache") # 'redis' or 'diskcache'
REDIS_URL = _env_str("REDIS_URL", "redis://localhost:6379/0")

# RAG Fusion
NUM_FUSION_QUERIES = _env_int("INVENIOAI_NUM_FUSION_QUERIES", default=2, min_value=1)