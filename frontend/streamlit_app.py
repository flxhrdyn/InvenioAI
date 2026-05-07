"""InvenioAI Streamlit chat UI.

This Streamlit app is a thin client for the FastAPI backend. Configure the
backend URL with `INVENIOAI_API_BASE_URL` (defaults to `http://localhost:8000`).
"""

import json
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

import importlib
import theme
importlib.reload(theme)
from theme import COLORS, CSS_VARS

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

# Active page tracking is now handled by Streamlit's multi-page system


# ── Design System ─────────────────────────────────────────────────────────────

# Inject Google Fonts
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown(f"""
<style>
{CSS_VARS}

/* ── Global Typography ── */
/* Apply font only to text-heavy elements to avoid breaking icons */
h1, h2, h3, h4, h5, h6, p, li, button, input, textarea, [data-testid="stChatMessage"] {{
    font-family: 'Outfit', sans-serif !important;
}}

/* ── Branding ── */
.branding {{
    font-weight: 700 !important;
}}

/* ── Welcome Screen Centering ── */
.welcome-container {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    min-height: 65vh; /* Adjusted to center in viewport */
}}

.welcome-title {{
    font-weight: 700 !important;
    font-size: 3.5rem !important;
    margin-bottom: 0.5rem !important;
}}

.welcome-subtitle {{
    font-size: 1.2rem !important;
    opacity: 0.8;
}}

/* ── Sidebar Spacing ── */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
    gap: 0.75rem !important;
}}

/* Tighten spacing between elements in the sidebar to match Dashboard */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
    gap: 0.75rem !important;
}}

/* ── Container Layout ── */
.block-container {{
    max-width: 1100px !important;
    padding-top: 2rem !important;
}}

/* ── UI Cleanup ── */
footer {{ visibility: hidden; }}
/* Keep header visible for the menu, but make it clean */
header {{ 
    background-color: transparent !important;
}}

/* ── Chat Message Styling (Boxed) ── */
[data-testid="stChatMessage"] {{
    border: 1px solid var(--invenio-border);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    background-color: transparent !important;
    transition: transform 0.2s ease, border-color 0.2s ease;
}}

[data-testid="stChatMessage"]:hover {{
    border-color: var(--invenio-accent);
}}

/* ── Expander Styling ── */
.stExpander {{
    border: 1px solid var(--invenio-border) !important;
    border-radius: 8px !important;
    background-color: var(--invenio-bg-card) !important;
}}

/* ── Status Bar Alignment ── */
.stStatusWidget {{
    border: 1px solid var(--invenio-border) !important;
    border-radius: 8px !important;
}}

/* ── Button Transitions ── */
button {{
    transition: all 0.2s ease !important;
}}
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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("InvenioAI")
    st.caption("AI · Document Intelligence")

    # Upload Section
    st.subheader("📤 Upload PDF")

    delete_after_index = (
        (os.getenv("INVENIOAI_DELETE_UPLOADED_PDFS") or os.getenv("DELETE_UPLOADED_PDFS") or "0").strip() == "1"
    )
    if delete_after_index:
        st.info('Uploaded PDFs will be deleted after indexing.')

    uploaded_file = st.file_uploader("Add documents to build your AI knowledge base", type=["pdf"])
    if uploaded_file:
        upload_status_slot = st.empty()
        if st.button("⚡ Process & Index", type="primary", use_container_width=True):
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



    # Knowledge Base
    st.subheader("🧠 Knowledge Base")
    indexed_files = get_indexed_files()
    if indexed_files:
        for f in indexed_files:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.write(f"📄 {f}")
            with col2:
                if st.button("🗑️", key=f"del_{f}"):
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
                        if hasattr(e, 'response') and e.response is not None:
                            try:
                                detail = e.response.json().get("detail", e.response.text)
                                st.error(f"❌ Delete Failed: {detail}")
                            except Exception:
                                st.error(f"❌ Delete Failed: {e.response.text}")
                        else:
                            st.error(f"❌ Failed: {e}")
    else:
        st.write("No documents yet.")
    # Force spacing to match Dashboard (cancel out list container padding)
    st.markdown('<div style="margin-top: -70px;"></div>', unsafe_allow_html=True)
    
    # Action Buttons (tightly coupled to Knowledge Base)
    if st.button("🗑️ Delete All Documents", use_container_width=True):
        with st.spinner("Deleting..."):
            try:
                resp = requests.delete(f"{API_BASE_URL}/documents", timeout=60)
                resp.raise_for_status()
                st.session_state.messages = []
                clear_persistent_history()
                _fetch_indexed_documents.clear()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed: {e}")
    
    if st.button("💬 Clear Chat History", use_container_width=True):
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
                <h1 class="welcome-title">🧠 InvenioAI</h1>
                <p class="welcome-subtitle">Ask anything about your Knowledge Base.</p>
            </div>
        """, unsafe_allow_html=True)

    # Render history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Display Thinking Process (if any)
            thoughts = message.get("thoughts")
            if thoughts:
                with st.expander("🧠 Thought Process", expanded=False):
                    if isinstance(thoughts, list):
                        # Join with newlines for list of steps
                        st.markdown("\n".join(thoughts))
                    else:
                        st.markdown(thoughts)
            
            st.markdown(message["content"])
        
            # Interactive Sources for assistant messages
            sources = message.get("sources")
            if sources and isinstance(sources, list):
                # Group sources by filename
                from collections import defaultdict
                grouped = defaultdict(list)
                for s in sources:
                    grouped[s['file']].append({
                        "text": s.get('text', ''),
                        "page": s.get('page')
                    })
                
                source_items = list(grouped.items())
                if source_items:
                    with st.expander(f"📚 {len(source_items)} Sources", expanded=False):
                        for filename, snippets in source_items:
                            st.markdown(f"**📄 {filename}**")
                            for item in snippets:
                                text = item["text"]
                                page = item["page"]
                                if page:
                                    st.caption(f"Page {page}")
                                st.markdown(f"> {text}")

def run_streaming_query(prompt: str, history: list[str]):
    """Consume the SSE stream from the backend and update the UI in real-time."""
    
    with st.chat_message("assistant"):
        thought_container = st.status("🧠 Thinking...", expanded=True)
        answer_placeholder = st.empty()
        full_answer = ""
        sources = []
        thoughts = []
        
        try:
            # Use stream=True to handle SSE
            response = requests.post(
                f"{API_BASE_URL}/query/stream",
                json={"question": prompt, "history": history},
                stream=True,
                timeout=(5, 60) # 5s to connect, 60s for the whole stream
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line:
                    continue
                
                line_str = line.decode("utf-8")
                if not line_str.startswith("data: "):
                    continue
                
                try:
                    data = json.loads(line_str[6:])
                    step = data.get("step")
                    
                    if step == "cached":
                        thought_container.update(label="⚡ Serving from cache...", state="running")
                    elif step == "rewriting":
                        thought_container.update(label="🔍 Rewriting query for context...")
                    elif step == "retrieving":
                        thought_container.update(label="🛰️ Searching document library...")
                    elif step == "reranking":
                        thought_container.update(label="🎯 Ranking relevant chunks...")
                    elif step == "generating":
                        thought_container.update(label="🧠 Synthesizing answer...")
                    elif step == "thinking":
                        content = data.get("content", "")
                        # We accumulate thoughts (ONLY CoT content)
                        thoughts.append(content)
                        
                        # Update loading status label with a snippet
                        snippet = content.strip().replace("\n", " ")[:80]
                        if snippet:
                            # Highlight Step transitions
                            if "Step " in snippet:
                                thought_container.update(label=f"⚙️ {snippet}")
                            else:
                                thought_container.update(label=f"🧠 {snippet}...")
                    elif step == "token":
                        content = data.get("content", "")
                        full_answer += content
                        # Collapse thoughts once we start generating heavily
                        thought_container.update(label="✅ Reasoning Complete", state="complete", expanded=False)
                        answer_placeholder.markdown(full_answer + " ▌")
                    elif step == "done":
                        full_answer = data.get("answer", full_answer)
                        sources = data.get("sources", [])
                        backend_thoughts = data.get("thoughts")
                        
                        # Store only the CoT content as thoughts
                        thoughts = backend_thoughts if backend_thoughts else "".join(thoughts)

                        answer_placeholder.markdown(full_answer)
                        thought_container.update(label="✅ Reasoning Complete", state="complete", expanded=False)
                    elif step == "error":
                        error_msg = data.get("message", "Unknown backend error")
                        st.error(f"❌ **Pipeline Error:** {error_msg}")
                        thought_container.update(label="❌ Error in Pipeline", state="error")
                        return None, [], []
                except json.JSONDecodeError:
                    continue
            
            return full_answer, sources, thoughts

        except Exception as e:
            st.error(f"❌ **Connection Error:** {e}")
            return None, [], []


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

        formatted_history = [
            f"{m['role']}: {m['content']}"
            for m in st.session_state.messages[:-1]
        ]

        answer, sources, thoughts = run_streaming_query(prompt, formatted_history)
        
        if answer:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources,
                "thoughts": thoughts
            })
            save_persistent_history(st.session_state.messages)
            st.rerun()
