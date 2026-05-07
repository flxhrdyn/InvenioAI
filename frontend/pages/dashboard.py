"""InvenioAI Streamlit dashboard.

Renders a lightweight analytics view based on the local `metrics.json` file
written by the backend.
"""
import math
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add project root to the import path so `app.*` and `frontend.*` imports work
# when Streamlit executes pages from this subdirectory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from frontend.theme import COLORS

from backend.app.metrics import (
    compute_ir_metrics,
    load_metrics,
    per_query_ir_metrics,
    reset_metrics,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InvenioAI · Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Mark this page as active so the chat page can cancel any in-flight UI updates.
st.session_state["invenioai_active_page"] = "dashboard"

# ── Design System (mirrors main app) ─────────────────────────────────────────

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor=COLORS["bg_card"],
        plot_bgcolor=COLORS["bg_secondary"],
        font=dict(family="Inter, sans-serif", color=COLORS["text_primary"]),
        xaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"]),
        yaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"]),
        legend=dict(bgcolor=COLORS["bg_card"], bordercolor=COLORS["border"]),
        margin=dict(l=16, r=16, t=32, b=16),
    )
)


def _layout_kwargs(*, drop: tuple[str, ...] = ()) -> dict:
    layout = PLOTLY_TEMPLATE["layout"].to_plotly_json()
    for key in drop:
        layout.pop(key, None)
    return layout

# ── Design System (Clean Defaults) ─────────────────────────────────────────────

st.markdown("""
<style>
.block-container {
    max-width: 1200px !important;
}
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(val: float, pct: bool = True) -> str:
    if val is None:
        return "—"
    return f"{val * 100:.1f}%" if pct else f"{val:.4f}"


def _color_for(val: float) -> str:
    if val is None or not math.isfinite(val):
        return COLORS["text_muted"]
    if val >= 0.75:
        return COLORS["success"]
    if val >= 0.5:
        return COLORS["warning"]
    return COLORS["error"]


# ── Sidebar ───────────────────────────────────────────────────────────────────
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("InvenioAI")
    st.caption("AI · Document Intelligence")

    st.subheader("Evaluation Parameters")

    k_val = st.slider("k (top-k docs)", min_value=1, max_value=10, value=5,
                      help="Number of top documents evaluated in each @k metric.")

    threshold = st.slider("Relevance Threshold", min_value=0.0, max_value=1.0, value=0.01, step=0.01,
                          help="Score ≥ threshold → relevant.")

    st.divider()
    st.subheader("Actions")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

    if st.button("🗑️ Reset Metrics", use_container_width=True):
        reset_metrics()
        st.success("Metrics reset!")
        st.rerun()

    st.divider()
    st.info("Relevance is estimated from cosine similarity scores.")


# ── Load data ─────────────────────────────────────────────────────────────────
raw = load_metrics()
history = raw.get("query_history", [])
ir_agg = compute_ir_metrics(history, k=k_val, threshold=threshold)
per_q   = per_query_ir_metrics(history, k=k_val, threshold=threshold)

evaluated = ir_agg["evaluated_queries"]
total_q   = raw.get("total_queries", 0)
total_docs_indexed = raw.get("total_documents_indexed", 0)

avg_resp = (raw.get("total_response_time", 0) / total_q) if total_q else 0
avg_ret  = (raw.get("total_retrieval_time", 0) / total_q) if total_q else 0
avg_gen  = (raw.get("total_generation_time", 0) / total_q) if total_q else 0
avg_docs = (raw.get("total_docs_retrieved", 0) / total_q) if total_q else 0

# ── Header ────────────────────────────────────────────────────────────────────
# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Retrieval Metrics Dashboard")
st.write(f"Evaluate RAG retrieval quality. **@k = {k_val}**, **threshold = {threshold:.2f}**. ({evaluated} / {total_q} queries evaluated)")

# ── No data guard ─────────────────────────────────────────────────────────────
if total_q == 0:
    st.info("📭 No queries recorded yet. Start asking questions on the Chat page.")
    st.stop()

if evaluated == 0:
    st.warning(
        "⚠️ No queries with retrieval scores found. "
        "New queries made after this update will save scores automatically. "
        "Try asking a question on the Chat page."
    )

# ── Core KPI cards (lean default view) ───────────────────────────────────────
# ── Core KPI cards (standard st.metric) ───────────────────────────────────────
st.subheader("📌 Core Quality & Performance")

core_cols = st.columns(5)
core_cols[0].metric("HitRate@k", _fmt(ir_agg["hit_rate"]), help="Queries with ≥1 relevant result")
core_cols[1].metric("nDCG@k", _fmt(ir_agg["ndcg"]), help="Ranking quality of top-k results")
core_cols[2].metric("Avg Response", f"{avg_resp:.2f}s", help="End-to-end response latency")
core_cols[3].metric("Avg Retrieval", f"{avg_ret:.2f}s", help="Retriever latency")
core_cols[4].metric("Indexed Docs", str(total_docs_indexed), help="Total PDFs indexed")


# ── Response time charts ──────────────────────────────────────────────────────
st.subheader("⏱️ Response Time Trend")

if per_q:
    df_time = pd.DataFrame(per_q).reset_index(drop=True)
    df_time["#"] = df_time.index + 1

    fig_time = go.Figure()
    time_series = [
        ("Total Time",      "response_time",  COLORS["chart1"]),
        ("Retrieval Time",  "retrieval_time", COLORS["chart2"]),
        ("Generation Time", "generation_time",COLORS["chart3"]),
    ]
    for name, col, color in time_series:
        fig_time.add_trace(go.Scatter(
            x=df_time["#"].tolist(),
            y=df_time[col].tolist(),
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color),
            hovertemplate=f"<b>{name}</b><br>Query #%{{x}}<br>%{{y:.2f}}s<extra></extra>",
        ))

    fig_time.update_layout(
        paper_bgcolor=COLORS["bg_card"],
        plot_bgcolor=COLORS["bg_secondary"],
        font=dict(family="Inter, sans-serif", color=COLORS["text_primary"]),
        margin=dict(l=16, r=16, t=32, b=16),
        height=280,
        xaxis=dict(
            title="Query #",
            gridcolor=COLORS["border"],
            linecolor=COLORS["border"],
            autorange=True,
        ),
        yaxis=dict(
            title="Time (seconds)",
            gridcolor=COLORS["border"],
            linecolor=COLORS["border"],
            autorange=True,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor=COLORS["bg_card"], bordercolor=COLORS["border"],
        ),
    )
    st.plotly_chart(fig_time, use_container_width=True, config={"displayModeBar": False})

# ── Aggregate stats ───────────────────────────────────────────────────────────
st.subheader("📊 Aggregate Statistics")

agg_cols = st.columns(4)
agg_cols[0].metric("Total Queries", str(total_q))
agg_cols[1].metric("Avg Response", f"{avg_resp:.2f}s")
agg_cols[2].metric("Avg Retrieval", f"{avg_ret:.2f}s")
agg_cols[3].metric("Avg Docs Retrieved", f"{avg_docs:.1f}")

# ── Per-query table ───────────────────────────────────────────────────────────
st.subheader("🗂️ Query History (Core)")

if per_q:
    df_table = pd.DataFrame(per_q)
    if not df_table.empty:
        df_table['timestamp'] = pd.to_datetime(df_table['timestamp'])
        df_table = df_table.sort_values('timestamp', ascending=False).reset_index(drop=True)

    display_cols = {
        "question":        "Question",
        "response_time":   "Resp (s)",
        "retrieval_time":  "Retr (s)",
        "docs_retrieved":  "Docs",
        "ndcg":            f"nDCG@{k_val}",
        "hit_rate":        f"HR@{k_val}",
    }
    df_display = df_table[list(display_cols.keys())].copy()
    df_display.columns = list(display_cols.values())

    # Format score columns
    for col in [f"nDCG@{k_val}", f"HR@{k_val}"]:
        df_display[col] = df_display[col].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "—"
        )

    st.dataframe(df_display, use_container_width=True, hide_index=True, height=320)
else:
    st.info("No query history yet.")


# ── Advanced retrieval metrics (optional) ────────────────────────────────────
with st.expander("Advanced Retrieval Metrics (optional)", expanded=False):
    st.markdown(
        f"""
        <div style="font-size:12px; color:{COLORS['text_secondary']}; margin-bottom:12px;">
            Advanced metrics use score-threshold relevance (threshold = {threshold:.2f})
            and are best used for retrieval debugging.
        </div>
        """,
        unsafe_allow_html=True,
    )

    adv_cols = st.columns(4)
    adv_cols[0].metric("Precision@k", _fmt(ir_agg["precision"]))
    adv_cols[1].metric("Recall@k", _fmt(ir_agg["recall"]))
    adv_cols[2].metric("MRR", _fmt(ir_agg["mrr"]))
    adv_cols[3].metric("Avg Gen Time", f"{avg_gen:.2f}s")

    st.subheader("📚 Metric Definitions")
    definitions = [
        ("Precision@k", COLORS["chart1"],
         "Of the top-k retrieved docs, how many are truly relevant?",
         r"$$\text{P@k} = \frac{|\{rel\} \cap \text{top-k}|}{k}$$"),
        ("Recall@k", COLORS["chart2"],
         "Of all relevant docs, how many were found within top-k?",
         r"$$\text{R@k} = \frac{|\{rel\} \cap \text{top-k}|}{|\{rel\}|}$$"),
        ("MRR", COLORS["chart3"],
         "How highly ranked is the first relevant document? Higher is better.",
         r"$$\text{MRR} = \frac{1}{|Q|}\sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$"),
    ]

    def_cols = st.columns(3)
    for col, (title, color, desc, formula) in zip(def_cols, definitions):
        with col:
            st.write(f"**{title}**")
            st.caption(desc)
            st.latex(formula.replace("$", ""))
