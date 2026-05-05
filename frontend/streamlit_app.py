"""InvenioAI Streamlit chat UI.

This Streamlit app is a thin client for the FastAPI backend. Configure the
backend URL with `INVENIOAI_API_BASE_URL` (defaults to `http://localhost:8000`).
"""

import os
import sys
import time

import requests
import streamlit as st
from pathlib import Path

# Add backend to path so we can use CacheManager
sys.path.append(str(Path(__file__).parent.parent / "backend"))
from app.cache_manager import CacheManager

# Singleton-like getter for CacheManager in frontend
@st.cache_resource
def get_frontend_cache():
    return CacheManager()

CHAT_HISTORY_CACHE_KEY = "invenio_persistent_chat_history"

def load_persistent_history():
    cache = get_frontend_cache()
    return cache.get(CHAT_HISTORY_CACHE_KEY) or []

def save_persistent_history(messages):
    cache = get_frontend_cache()
    cache.set(CHAT_HISTORY_CACHE_KEY, messages)

def clear_persistent_history():
    cache = get_frontend_cache()
    cache.set(CHAT_HISTORY_CACHE_KEY, None)

st.set_page_config(
    page_title="InvenioAI | Intelligent RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from theme import COLORS

API_BASE_URL = os.getenv("INVENIOAI_API_BASE_URL", "http://localhost:8000").rstrip("/")



def _is_hf_spaces_runtime() -> bool:
    return bool(os.getenv("SPACE_ID") or os.getenv("SPACE_HOST"))


def _get_upload_timeout_seconds() -> int:
    # Upload + indexing may take longer on cold starts, especially in HF Spaces.
    default_timeout = 600 if _is_hf_spaces_runtime() else 120
    raw = (os.getenv("INVENIOAI_UPLOAD_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return default_timeout

    try:
        value = int(raw)
    except ValueError:
        return default_timeout

    return max(30, min(value, 3600))


UPLOAD_TIMEOUT_SECONDS = _get_upload_timeout_seconds()
UPLOAD_JOB_POLL_INTERVAL_SECONDS = 1.0
UPLOAD_JOB_WAIT_SECONDS = UPLOAD_TIMEOUT_SECONDS
UPLOAD_DURATION_HISTORY_KEY = "upload_duration_history"
MAX_UPLOAD_DURATION_SAMPLES = 10


def _get_assistant_typing_enabled() -> bool:
    raw = (os.getenv("INVENIOAI_ASSISTANT_TYPING_EFFECT") or "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _get_assistant_word_delay_seconds() -> float:
    raw = (os.getenv("INVENIOAI_ASSISTANT_TYPING_WORD_DELAY_SECONDS") or "0.016").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 0.016
    return max(0.0, min(value, 0.08))


def _get_assistant_typing_max_words() -> int:
    raw = (os.getenv("INVENIOAI_ASSISTANT_TYPING_MAX_WORDS") or "140").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 140
    return max(20, min(value, 500))


ASSISTANT_TYPING_ENABLED = _get_assistant_typing_enabled()
ASSISTANT_TYPING_WORD_DELAY_SECONDS = _get_assistant_word_delay_seconds()
ASSISTANT_TYPING_MAX_WORDS = _get_assistant_typing_max_words()


def _is_chat_active() -> bool:
    return True


def _get_upload_duration_history() -> list[float]:
    history = st.session_state.get(UPLOAD_DURATION_HISTORY_KEY)
    if isinstance(history, list):
        clean: list[float] = []
        for value in history:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric > 0:
                clean.append(numeric)
        st.session_state[UPLOAD_DURATION_HISTORY_KEY] = clean[-MAX_UPLOAD_DURATION_SAMPLES:]
        return st.session_state[UPLOAD_DURATION_HISTORY_KEY]

    st.session_state[UPLOAD_DURATION_HISTORY_KEY] = []
    return st.session_state[UPLOAD_DURATION_HISTORY_KEY]


def _record_upload_duration(seconds: float) -> None:
    if seconds <= 0:
        return
    history = _get_upload_duration_history()
    history.append(seconds)
    st.session_state[UPLOAD_DURATION_HISTORY_KEY] = history[-MAX_UPLOAD_DURATION_SAMPLES:]


def fetch_metrics() -> tuple[dict | None, str | None]:
    """Fetch aggregate metrics from backend."""
    try:
        resp = requests.get(f"{API_BASE_URL}/metrics", timeout=10)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"❌ **Error {resp.status_code}:** Failed to load metrics."
    except Exception as e:
        return None, f"❌ **Connection Error:** {e}"


def _estimate_upload_eta_seconds(elapsed_seconds: float) -> int | None:
    history = _get_upload_duration_history()
    if not history:
        return None

    avg_seconds = sum(history) / len(history)
    remaining = int(round(avg_seconds - elapsed_seconds))
    if remaining <= 0:
        return 0
    return remaining


def _render_assistant_message(reply: str) -> None:
    """Render assistant reply with a safe typing effect.

    We animate short previews for UX, then always render the final full markdown
    atomically to avoid broken layout after page switches/reruns.
    """
    if not ASSISTANT_TYPING_ENABLED:
        st.markdown(reply)
        return

    words = reply.split()
    if not words:
        st.markdown(reply)
        return

    preview_words = min(len(words), ASSISTANT_TYPING_MAX_WORDS)
    placeholder = st.empty()
    current: list[str] = []

    for word in words[:preview_words]:
        if not _is_chat_active():
            break
        current.append(word)
        placeholder.markdown(" ".join(current) + " ▌")
        time.sleep(ASSISTANT_TYPING_WORD_DELAY_SECONDS)

    placeholder.markdown(reply)

st.set_page_config(
    page_title="InvenioAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Active page tracking is now handled by Streamlit's multi-page system


# ── Design System ─────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
/* ── Global reset ── */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {COLORS['bg_primary']};
    color: {COLORS['text_primary']};
}}

/* ── Center main content area in wide layout ── */
.block-container {{
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}}

/* ── App background ── */
.stApp {{
    background-color: {COLORS['bg_primary']};
}}

/* ── Hide Streamlit default header/footer ── */
#MainMenu, footer, header {{ visibility: hidden; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {COLORS['bg_sidebar']};
    border-right: 1px solid {COLORS['border']};
}}
[data-testid="stSidebar"] * {{
    color: {COLORS['text_primary']};
}}

/* ── Sidebar header branding ── */
.brand-header {{
    padding: 16px 0;
    border-bottom: 1px solid {COLORS['border']};
    margin-bottom: 20px;
}}
.brand-title {{
    font-size: 22px;
    font-weight: 700;
    color: {COLORS['text_primary']};
    letter-spacing: -0.3px;
}}
.brand-title span {{
    color: {COLORS['accent']};
}}
.brand-subtitle {{
    font-size: 11px;
    font-weight: 400;
    color: {COLORS['text_muted']};
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-top: 4px;
}}

/* ── Section labels ── */
.section-label {{
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: {COLORS['text_muted']};
    margin-bottom: 12px;
    margin-top: 8px;
}}

/* ── Simple Doc Pill ── */
.doc-pill {{
    display: flex;
    align-items: center;
    gap: 10px;
    background: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    color: {COLORS['text_primary']};
    width: 100%;
    margin-bottom: 4px;
}}
.doc-pill-icon {{
    color: {COLORS['accent']};
    flex-shrink: 0;
}}
.doc-pill-text {{
    flex-grow: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}
.delete-btn-container {{
    display: flex;
    align-items: center;
    justify-content: center;
}}

/* ── Empty state ── */
.empty-state {{
    text-align: center;
    padding: 20px;
    background: {COLORS['bg_card']};
    border: 1px dashed {COLORS['border']};
    border-radius: 10px;
    color: {COLORS['text_muted']};
    font-size: 13px;
}}

/* ── Metric cards ── */
.metric-card {{
    background: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    margin-bottom: 8px;
}}
.metric-value {{
    font-size: 22px;
    font-weight: 700;
    color: {COLORS['accent']};
    line-height: 1.2;
}}
.metric-label {{
    font-size: 11px;
    color: {COLORS['text_muted']};
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* ── Buttons ── */
.stButton > button {{
    background: {COLORS['accent']};
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 13px;
    padding: 8px 16px;
    transition: background 0.2s ease;
    width: 100%;
}}
.stButton > button:hover {{
    background: {COLORS['accent_hover']};
    color: #ffffff;
    border: none;
}}
.stButton > button[kind="secondary"] {{
    background: {COLORS['bg_card']};
    color: {COLORS['text_secondary']};
    border: 1px solid {COLORS['border']};
}}
.stButton > button[kind="secondary"]:hover {{
    background: {COLORS['border']};
    color: {COLORS['text_primary']};
}}

/* ── Minimalist Delete Link ── */
.delete-action-row {{
    display: flex;
    justify-content: flex-end;
    margin-top: -8px;
    margin-bottom: 12px;
}}
.small-delete-btn button {{
    background: transparent !important;
    color: {COLORS['text_muted']} !important;
    border: none !important;
    font-size: 11px !important;
    padding: 0 !important;
    margin: 0 !important;
    height: auto !important;
    text-decoration: none !important;
}}
.small-delete-btn button:hover {{
    color: #ff4b4b !important;
    text-decoration: underline !important;
    background: transparent !important;
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    background: {COLORS['bg_card']};
    border: 1.5px dashed {COLORS['border']};
    border-radius: 10px;
    padding: 10px;
}}
[data-testid="stFileUploader"] label {{
    color: {COLORS['text_secondary']} !important;
    font-size: 13px;
}}

/* ── Aggressive Wide Layout Fix ── */
.block-container {{
    max-width: 1200px !important;
    width: 100% !important;
    margin: 0 auto !important;
}}

/* ── Chat input (Force Wide Rectangular) ── */
[data-testid="stChatInput"] {{
    background-color: transparent !important;
    padding: 24px 20px 40px !important;
    width: 100% !important;
}}
[data-testid="stChatInput"] > div {{
    width: 100% !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}}
[data-testid="stChatInput"] form {{
    background: {COLORS['bg_card']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 16px !important; 
    padding: 8px 20px !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3) !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
}}
[data-testid="stChatInput"] textarea {{
    background-color: transparent !important;
    border: none !important;
    color: {COLORS['text_primary']} !important;
    font-size: 15px !important;
    font-family: 'Inter', sans-serif !important;
    padding: 12px 0 !important;
    box-shadow: none !important;
    line-height: 1.5 !important;
    flex-grow: 1 !important;
}}
[data-testid="stChatInput"] button {{
    background: {COLORS['accent']} !important;
    width: 40px !important;
    height: 40px !important;
    min-width: 40px !important;
    border-radius: 10px !important;
    border: none !important;
    margin-left: 12px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex-shrink: 0 !important;
}}

/* ── Chat message base ── */
[data-testid="stChatMessage"] {{
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin-bottom: 24px !important;
    align-items: flex-start;
}}

/* Hide Streamlit's default empty username/name labels inside chat contents */
[data-testid="stChatMessageContent"] > div:first-child:not(:has(div)) {{
    display: none !important;
}}

/* ── Chat messages — user (right) ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
    flex-direction: row-reverse !important;
    align-items: flex-start !important;
    gap: 16px !important;
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {{
    background: {COLORS['accent']} !important;
    border-radius: 20px 4px 20px 20px !important;
    padding: 10px 18px !important;
    width: fit-content !important;
    max-width: min(820px, 80%) !important;
    margin-top: 2px !important;
    margin-left: auto !important;
    margin-right: 0 !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(108,99,255,0.35) !important;
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] *,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] > div,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] > div > div,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] .stMarkdown,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] {{
    margin: 0 !important;
    padding: 0 !important;
    margin-block-start: 0 !important;
    margin-block-end: 0 !important;
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] li,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] span {{
    color: rgba(255,255,255,0.95) !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    margin: 0 !important;
    margin-block-start: 0 !important;
    margin-block-end: 0 !important;
    word-break: break-word;
}}

/* ── Chat messages — assistant (left) ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {{
    align-items: flex-start !important;
    gap: 16px !important;
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {{
    background: {COLORS['bg_card']} !important;
    border-radius: 4px 20px 20px 20px !important;
    padding: 10px 18px !important;
    width: fit-content !important;
    max-width: min(1000px, 90%) !important;
    margin-top: 2px !important;
    margin-left: 0 !important;
    margin-right: auto !important;
    border: 1px solid {COLORS['border']} !important;
    border-left: 3px solid {COLORS['accent']} !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.25) !important;
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] *,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] > div,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] > div > div,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] .stMarkdown,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] {{
    margin: 0 !important;
    padding: 0 !important;
    margin-block-start: 0 !important;
    margin-block-end: 0 !important;
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] li {{
    color: {COLORS['text_primary']} !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    margin: 0 !important;
    padding: 0 !important;
    margin-block-start: 0 !important;
    margin-block-end: 0 !important;
    word-break: break-word;
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] p + p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] li + li {{
    margin-top: 6px !important;
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] strong {{
    color: {COLORS['text_primary']} !important;
    font-weight: 600;
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] code {{
    background: {COLORS['bg_secondary']} !important;
    color: {COLORS['accent']} !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-size: 13px !important;
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] hr {{
    border-color: {COLORS['border']} !important;
    margin: 10px 0 !important;
}}

/* ── Avatar icons ── */
[data-testid="chatAvatarIcon-user"] {{
    background: {COLORS['accent']} !important;
    border: none !important;
    flex-shrink: 0;
    margin: 0 !important;
}}
[data-testid="chatAvatarIcon-assistant"] {{
    background: {COLORS['accent']} !important;
    border: none !important;
    flex-shrink: 0;
    margin: 0 !important;
}}

/* ── Welcome screen ── */
.welcome-container {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 48px 24px;
    text-align: center;
}}
.welcome-icon {{
    font-size: 52px;
    margin-bottom: 16px;
    filter: drop-shadow(0 0 20px rgba(108,99,255,0.5));
}}
.welcome-title {{
    font-size: 26px;
    font-weight: 700;
    color: {COLORS['text_primary']};
    margin-bottom: 12px;
    letter-spacing: -0.5px;
}}
.welcome-subtitle {{
    font-size: 15px;
    color: {COLORS['text_secondary']};
    max-width: 460px;
    line-height: 1.6;
}}

/* ── Spinner ── */
[data-testid="stSpinner"] {{
    color: {COLORS['accent']} !important;
}}

/* ── Divider ── */
hr {{
    border-color: {COLORS['border']} !important;
    margin: 16px 0 !important;
}}

/* ── Expander ── */
[data-testid="stExpander"] {{
    background: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
}}
[data-testid="stExpander"] summary {{
    color: {COLORS['text_secondary']};
    font-size: 13px;
    font-weight: 500;
}}

/* ── Alerts ── */
[data-testid="stAlert"] {{
    border-radius: 10px;
    border: none;
    font-size: 13px;
}}
[data-testid="stAlert"] p {{
    margin: 0 !important;
}}
[data-testid="stChatMessageContent"] [data-testid="stAlert"] {{
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
}}
[data-testid="stChatMessageContent"] [data-testid="stAlert"] > div {{
    padding: 0 !important;
}}
[data-testid="stChatMessageContent"] [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {{
    display:flex;
    align-items: center;
}}

/* ── DataFrame ── */
[data-testid="stDataFrame"] {{
    border-radius: 8px;
    overflow: hidden;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {COLORS['bg_secondary']}; }}
::-webkit-scrollbar-thumb {{ background: {COLORS['border']}; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: {COLORS['accent']}; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_indexed_documents(api_base_url: str) -> list[str]:
    try:
        resp = requests.get(f"{api_base_url}/documents", timeout=5)
        if resp.status_code != 200:
            return []
        payload = resp.json() or {}
        # The backend returns a list of filenames directly based on my recent checks
        if isinstance(payload, list):
            return [str(d) for d in payload if d]
        docs = payload.get("documents") or []
        return [str(d) for d in docs if d]
    except Exception:
        return []


def get_indexed_files() -> list[str]:
    # Prefer backend source-of-truth (Qdrant metadata).
    try:
        docs = _fetch_indexed_documents(API_BASE_URL)
        if docs:
            return docs
    except Exception:
        pass

    return []


def format_error_message(response: requests.Response) -> str:
    text = response.text.lower()
    if "quota" in text or "rate_limit" in text or "429" in text:
        return (
            "⚠️ **Groq API Rate Limit exceeded.** "
            "You have reached the request limit for your Groq plan. "
            "Wait a moment or check your Groq dashboard."
        )
    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = response.text
    return f"❌ **Error {response.status_code}:** {detail}"


def create_upload_job(uploaded_file) -> tuple[str | None, str | None]:
    try:
        resp = requests.post(
            f"{API_BASE_URL}/upload/jobs",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
            timeout=120,
        )
    except requests.exceptions.ConnectionError:
        return None, f"❌ **Connection Error:** Could not connect to the backend at {API_BASE_URL}."
    except requests.exceptions.Timeout:
        return None, "⏱️ **Timeout:** Upload request took too long. Please try again."

    if resp.status_code != 200:
        return None, format_error_message(resp)

    try:
        payload = resp.json()
    except Exception:
        return None, f"❌ **Error {resp.status_code}:** {resp.text}"

    job_id = payload.get("job_id")
    if not job_id:
        return None, "❌ **Error:** Missing upload job id from backend response."
    return str(job_id), None


def fetch_upload_job(job_id: str) -> tuple[dict | None, str | None]:
    try:
        resp = requests.get(f"{API_BASE_URL}/upload/jobs/{job_id}", timeout=15)
    except requests.exceptions.ConnectionError:
        return None, f"❌ **Connection Error:** Could not connect to the backend at {API_BASE_URL}."
    except requests.exceptions.Timeout:
        return None, "⏱️ **Timeout:** Checking upload job took too long."

    if resp.status_code != 200:
        return None, format_error_message(resp)

    try:
        return resp.json(), None
    except Exception:
        return None, f"❌ **Error {resp.status_code}:** {resp.text}"


def _render_upload_job_status(
    status_slot,
    status: str,
    filename: str,
    elapsed_seconds: float,
    eta_seconds: int | None,
) -> None:
    if status_slot is None:
        return

    label = f"**{filename}**" if filename else "**Document**"
    eta_suffix = ""
    if eta_seconds is not None:
        eta_suffix = " (ETA now)" if eta_seconds == 0 else f" (ETA ~{eta_seconds}s)"

    if status == "pending":
        status_slot.info(f"{label} queued for indexing... ({elapsed_seconds:.0f}s){eta_suffix}")
        return
    if status == "running":
        status_slot.info(f"{label} is being indexed... ({elapsed_seconds:.0f}s){eta_suffix}")
        return
    if status == "succeeded":
        status_slot.success(f"{label} indexing completed.")
        return
    if status == "failed":
        status_slot.error(f"{label} indexing failed.")
        return

    status_slot.info(f"{label} status: {status}")


def wait_for_upload_job(job_id: str, *, status_slot=None, filename: str = "") -> tuple[bool, str]:
    started = time.monotonic()
    while True:
        elapsed = time.monotonic() - started
        eta_seconds = _estimate_upload_eta_seconds(elapsed)
        job, err = fetch_upload_job(job_id)
        if err:
            if status_slot is not None:
                status_slot.error(err)
            return False, err

        status = (job or {}).get("status")
        _render_upload_job_status(status_slot, str(status), filename, elapsed, eta_seconds)

        if status == "succeeded":
            _record_upload_duration(elapsed)
            result = (job or {}).get("result") or {}
            done_filename = result.get("filename") or filename or "file"
            return True, f"✅ {done_filename} indexed successfully!"

        if status == "failed":
            error_msg = (job or {}).get("error") or "Unknown error"
            return False, f"❌ **Indexing failed:** {error_msg}"

        if time.monotonic() - started > UPLOAD_JOB_WAIT_SECONDS:
            return False, (
                "⏱️ Indexing is still running in the background. "
                "Please wait a moment, then click Refresh Data or reopen this page."
            )

        time.sleep(UPLOAD_JOB_POLL_INTERVAL_SECONDS)


def create_query_job(question: str, history: list[str]) -> tuple[str | None, str | None]:
    try:
        resp = requests.post(
            f"{API_BASE_URL}/query/jobs",
            json={"question": question, "history": history},
            timeout=30,
        )
    except requests.exceptions.ConnectionError:
        return None, f"❌ **Connection Error:** Could not connect to the backend at {API_BASE_URL}."
    except requests.exceptions.Timeout:
        return None, "⏱️ **Timeout:** The request took too long. Please try again."

    if resp.status_code != 200:
        return None, format_error_message(resp)

    try:
        payload = resp.json()
        return payload.get("job_id"), None
    except Exception:
        return None, f"❌ **Error {resp.status_code}:** {resp.text}"


def fetch_query_job(job_id: str) -> tuple[dict | None, str | None]:
    try:
        resp = requests.get(f"{API_BASE_URL}/query/jobs/{job_id}", timeout=15)
    except requests.exceptions.ConnectionError:
        return None, f"❌ **Connection Error:** Could not connect to the backend at {API_BASE_URL}."
    except requests.exceptions.Timeout:
        return None, "⏱️ **Timeout:** The request took too long. Please try again."

    if resp.status_code != 200:
        return None, format_error_message(resp)

    try:
        return resp.json(), None
    except Exception:
        return None, f"❌ **Error {resp.status_code}:** {resp.text}"


def build_reply_from_job_result(job: dict) -> tuple[str, list]:
    result = (job or {}).get("result") or {}
    answer = result.get("answer", "❌ **Error:** Empty response.")
    sources = result.get("sources", [])
    return answer, sources


def maybe_resume_pending_job():
    job_id = st.session_state.get("pending_job_id")
    prompt = st.session_state.get("pending_job_prompt")
    if not job_id or not prompt:
        return

    # If user is not on the chat page, don't update UI.
    # Keep the pending job so it can be resumed later.
    if not _is_chat_active():
        return

    with st.spinner("Searching for an answer..."):
        started = time.monotonic()
        while True:
            if not _is_chat_active():
                # User navigated away; stop sending UI updates.
                return

            job, err = fetch_query_job(job_id)
            if err:
                reply = err
                break

            status = (job or {}).get("status")
            if status in {"pending", "running"}:
                if time.monotonic() - started > 60:
                    # Keep job id so user can come back later without losing progress
                    return
                time.sleep(0.4) # Faster polling for better responsiveness
                continue

            if status == "succeeded":
                answer, sources = build_reply_from_job_result(job)
                
                # Render the typing effect for the new answer
                with st.chat_message("assistant"):
                    _render_assistant_message(answer)
                
                # Only then commit to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
                save_persistent_history(st.session_state.messages)
                st.session_state.pop("pending_job_id", None)
                st.session_state.pop("pending_job_prompt", None)
                st.rerun()

            if status == "failed":
                error_msg = (job or {}).get("error") or "Unknown error"
                reply = f"❌ **Error:** {error_msg}"
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.session_state.pop("pending_job_id", None)
                st.session_state.pop("pending_job_prompt", None)
                st.rerun()

            reply = f"❌ **Error:** Unknown job status: {status}"
            break

    st.session_state.pop("pending_job_id", None)
    st.session_state.pop("pending_job_prompt", None)

    if not _is_chat_active():
        # User navigated away right after completion.
        return

    with st.chat_message("assistant"):
        _render_assistant_message(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-header">
        <div class="brand-title">Invenio<span>AI</span></div>
        <div class="brand-subtitle">AI · Document Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # Upload
    st.markdown('<div class="section-label">Upload Document</div>', unsafe_allow_html=True)

    delete_after_index = (
        (os.getenv("INVENIOAI_DELETE_UPLOADED_PDFS") or os.getenv("DELETE_UPLOADED_PDFS") or "0").strip() == "1"
    )
    if delete_after_index:
        st.markdown(
            f'<div style="font-size:12px;color:{COLORS["text_muted"]};margin-top:-6px;margin-bottom:10px;">'
            'ℹ️ Uploaded PDFs will be deleted after indexing.'
            '</div>',
            unsafe_allow_html=True,
        )

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded_file:
        upload_status_slot = st.empty()
        if st.button("⚡ Index Document", type="primary", use_container_width=True):
            with st.spinner("Indexing document..."):
                job_id, err = create_upload_job(uploaded_file)
                if err or not job_id:
                    st.error(err or "❌ **Error:** Failed to create upload job.")
                else:
                    upload_status_slot.info(f"**{uploaded_file.name}** uploaded. Starting indexing job...")
                    ok, message = wait_for_upload_job(
                        job_id,
                        status_slot=upload_status_slot,
                        filename=uploaded_file.name,
                    )
                    if ok:
                        st.success(message)
                        _fetch_indexed_documents.clear()
                        st.rerun()
                    else:
                        if "still running in the background" in message:
                            st.info(message)
                        else:
                            st.error(message)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Indexed docs
    st.markdown('<div class="section-label">Indexed Documents</div>', unsafe_allow_html=True)
    indexed = get_indexed_files()
    if indexed:
        for f in indexed:
            name = f[:-4] if f.endswith(".pdf") else f
            # Document name pill
            st.markdown(
                f'''<div class="doc-pill">
                <span class="doc-pill-icon">📄</span>
                <span class="doc-pill-text" title="{f}">{name}</span>
                </div>''',
                unsafe_allow_html=True,
            )
            
            # Delete action below, aligned to the right
            cols = st.columns([0.6, 0.4])
            with cols[1]:
                st.markdown('<div class="small-delete-btn">', unsafe_allow_html=True)
                if st.button(f"🗑️ Delete", key=f"del_{f}"):
                    try:
                        resp = requests.delete(
                            f"{API_BASE_URL}/documents/delete", 
                            params={"filename": f},
                            timeout=30
                        )
                        resp.raise_for_status()
                        _fetch_indexed_documents.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="empty-state">📭 No documents yet.<br>Upload a PDF to get started.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Actions
    st.markdown('<div class="section-label">Actions</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Delete Docs", use_container_width=True):
            with st.spinner("Deleting..."):
                try:
                    resp = requests.delete(f"{API_BASE_URL}/documents", timeout=60)
                    resp.raise_for_status()
                    st.session_state.messages = []
                    clear_persistent_history()
                    _fetch_indexed_documents.clear()
                    st.rerun()
                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect.")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Delete timed out. Please try again.")
                except requests.exceptions.HTTPError as exc:
                    st.error(f"❌ {exc.response.text if exc.response else str(exc)}")
                except requests.exceptions.RequestException as exc:
                    st.error(f"❌ Delete failed: {exc}")
    with col2:
        if st.button("💬 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            clear_persistent_history()
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    


# ── Chat ──────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = load_persistent_history()

# Welcome screen when no messages
if _is_chat_active():
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-icon">🧠</div>
            <div class="welcome-title">InvenioAI</div>
            <div class="welcome-subtitle">
                Ask anything about the documents you have uploaded.
                The AI will find answers directly from your document sources.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Render history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
        # Interactive Source Pills for assistant messages
        sources = message.get("sources")
        if sources and isinstance(sources, list):
            st.markdown('<div style="margin-top: 16px; margin-bottom: 12px;"></div>', unsafe_allow_html=True)
            
            # Group sources by filename to avoid overwhelming redundancy
            from collections import defaultdict
            grouped = defaultdict(list)
            for s in sources:
                # Store snippet with its page info
                grouped[s['file']].append({
                    "text": s.get('text', ''),
                    "page": s.get('page')
                })
            
            # Create a clean single expander for all sources using native HTML <details> for jitter-free UI
            source_items = list(grouped.items())
            if source_items:
                src_html = ""
                for filename, snippets in source_items:
                    src_html += f'<div style="font-size: 13px; font-weight: 600; color: {COLORS["accent"]}; margin-bottom: 8px; margin-top: 12px;">📄 {filename}</div>'
                    for item in snippets:
                        text = item["text"]
                        page = item["page"]
                        page_label = f'<span style="font-size: 10px; background: {COLORS["bg_secondary"]}; padding: 2px 6px; border-radius: 4px; margin-bottom: 4px; display: inline-block;">Page {page}</span>' if page else ""
                        
                        # Truncate very long snippets to keep UI snappy
                        safe_text = (text[:800] + "...") if len(text) > 850 else text
                        src_html += f'''
                        <div style="margin-bottom: 12px; padding-left: 12px; border-left: 2px solid {COLORS["border"]};">
                            {page_label}
                            <div style="font-size: 12px; line-height: 1.5; color: #bbb;">{safe_text}</div>
                        </div>'''
                
                # Using native HTML <details> avoids Streamlit's layout engine jumping during interaction
                st.markdown(f'''
                <details class="source-details" style="border: 1px solid {COLORS["border"]}; border-radius: 8px; background: {COLORS["bg_card"]}; transition: all 0.2s ease; width: 100%; margin-bottom: 32px; margin-top: 8px;">
                    <summary style="padding: 10px 14px; color: {COLORS["text_secondary"]}; font-size: 13px; font-weight: 500; cursor: pointer; list-style: none; display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 14px;">📚</span> {len(source_items)} Sources
                        </div>
                        <svg class="chevron" viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="transition: transform 0.2s ease;">
                            <polyline points="6 9 12 15 18 9"></polyline>
                        </svg>
                    </summary>
                    <div style="padding: 0 14px 14px; max-height: 350px; overflow-y: auto; border-top: 1px solid {COLORS["border"]};">
                        {src_html}
                    </div>
                </details>
                <style>
                    .source-details summary::-webkit-details-marker {{ display: none; }}
                    .source-details[open] {{ border-color: {COLORS["accent"]}; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }}
                    .source-details[open] .chevron {{ transform: rotate(180deg); }}
                </style>
                ''', unsafe_allow_html=True)

# If user navigated away mid-request, resume polling and render the result.
maybe_resume_pending_job()

# Input
if prompt := st.chat_input("Ask something about your documents..."):
    if not get_indexed_files():
        with st.chat_message("assistant"):
            st.warning("⚠️ No documents indexed yet. Please upload a PDF in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_persistent_history(st.session_state.messages)
        with st.chat_message("user"):
            st.markdown(prompt)

        # Create a background job so the backend keeps working even if user navigates away.
        formatted_history = [
            f"{m['role']}: {m['content']}"
            for m in st.session_state.messages[:-1]
        ]

        job_id, err = create_query_job(prompt, formatted_history)
        if err or not job_id:
            reply = err or "❌ **Error:** Failed to create job."
            with st.chat_message("assistant"):
                _render_assistant_message(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            save_persistent_history(st.session_state.messages)
        else:
            st.session_state["pending_job_id"] = job_id
            st.session_state["pending_job_prompt"] = prompt
            maybe_resume_pending_job()


