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

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {COLORS['bg_primary']};
    color: {COLORS['text_primary']};
}}
.stApp {{ background-color: {COLORS['bg_primary']}; }}
#MainMenu, footer, header {{ visibility: hidden; }}

[data-testid="stSidebar"] {{
    background-color: {COLORS['bg_sidebar']};
    border-right: 1px solid {COLORS['border']};
}}
[data-testid="stSidebar"] * {{ color: {COLORS['text_primary']}; }}

/* Brand */
.brand-header {{
    padding: 8px 0 16px 0;
    border-bottom: 1px solid {COLORS['border']};
    margin-bottom: 20px;
}}
.brand-title {{ font-size:22px; font-weight:700; color:{COLORS['text_primary']}; letter-spacing:-0.3px; }}
.brand-title span {{ color:{COLORS['accent']}; }}
.brand-subtitle {{ font-size:11px; font-weight:400; color:{COLORS['text_muted']}; letter-spacing:0.8px; text-transform:uppercase; margin-top:2px; }}

/* Section label */
.section-label {{
    font-size:10px; font-weight:600; letter-spacing:1.2px;
    text-transform:uppercase; color:{COLORS['text_muted']};
    margin-bottom:10px; margin-top:4px;
}}

/* KPI card */
.kpi-card {{
    background: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 20px 18px;
    text-align: center;
    margin-bottom: 8px;
}}
.kpi-value {{
    font-size: 30px;
    font-weight: 700;
    color: {COLORS['accent']};
    line-height: 1.1;
}}
.kpi-label {{
    font-size: 11px;
    color: {COLORS['text_muted']};
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}}
.kpi-sub {{
    font-size: 11px;
    color: {COLORS['text_secondary']};
    margin-top: 2px;
}}

/* Section title */
.section-title {{
    font-size:15px; font-weight:600; color:{COLORS['text_primary']};
    margin: 24px 0 12px 0; letter-spacing: -0.2px;
}}

/* Info chip */
.info-chip {{
    display: inline-block;
    background: {COLORS['accent_light']};
    border: 1px solid {COLORS['accent']};
    border-radius: 20px;
    color: {COLORS['accent']};
    font-size: 11px; font-weight: 600;
    padding: 2px 10px;
    margin: 2px;
}}

/* Buttons */
.stButton > button {{
    background: {COLORS['accent']};
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 13px;
    padding: 8px 16px;
    transition: background 0.2s;
    width: 100%;
}}
.stButton > button:hover {{
    background: {COLORS['accent_hover']};
    color: #fff; border: none;
}}

/* Slider */
[data-testid="stSlider"] > div > div > div {{ background: {COLORS['accent']} !important; }}

/* DataFrame */
[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}

/* Divider */
hr {{ border-color: {COLORS['border']} !important; margin: 16px 0 !important; }}

/* Scrollbar */
::-webkit-scrollbar {{ width:4px; height:4px; }}
::-webkit-scrollbar-track {{ background:{COLORS['bg_secondary']}; }}
::-webkit-scrollbar-thumb {{ background:{COLORS['border']}; border-radius:4px; }}
::-webkit-scrollbar-thumb:hover {{ background:{COLORS['accent']}; }}
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
with st.sidebar:
    st.markdown("""
    <div class="brand-header">
        <div class="brand-title">Invenio<span>AI</span></div>
        <div class="brand-subtitle">AI · Document Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Evaluation Parameters</div>', unsafe_allow_html=True)

    k_val = st.slider("k  (top-k documents)", min_value=1, max_value=10, value=5,
                      help="Number of top documents evaluated in each @k metric.")

    threshold = st.slider("Relevance Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                          help="Score ≥ threshold → document is considered relevant (for Precision, Recall, MRR, HitRate).")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Actions</div>', unsafe_allow_html=True)

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

    if st.button("🗑️ Reset Metrics", use_container_width=True):
        reset_metrics()
        st.success("Metrics reset!")
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:11px; color:{COLORS['text_muted']}; line-height:1.7;">
        <b style="color:{COLORS['text_secondary']};">About Metrics</b><br>
        Relevance is estimated from the cosine similarity score between the query and document chunks.
        Scores ≥ threshold are considered relevant.
    </div>
    """, unsafe_allow_html=True)


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
st.markdown(f"""
<div style="padding:8px 0 24px 0;">
    <h1 style="font-size:26px; font-weight:700; color:{COLORS['text_primary']}; margin:0; letter-spacing:-0.5px;">
        📊 Retrieval Metrics Dashboard
    </h1>
    <p style="font-size:14px; color:{COLORS['text_secondary']}; margin-top:8px; margin-bottom:0;">
        Evaluate RAG retrieval quality using standard Information Retrieval metrics.
        &nbsp;
        <span class="info-chip">@k = {k_val}</span>
        <span class="info-chip">threshold = {threshold:.2f}</span>
        <span class="info-chip">{evaluated} / {total_q} queries evaluated</span>
    </p>
</div>
""", unsafe_allow_html=True)

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
st.markdown('<div class="section-title">📌 Core Quality & Performance</div>', unsafe_allow_html=True)

core_metrics = [
    ("HitRate@k", _fmt(ir_agg["hit_rate"]), _color_for(ir_agg["hit_rate"]) if evaluated > 0 else COLORS["text_muted"],
     "Queries with ≥1 relevant result"),
    ("nDCG@k", _fmt(ir_agg["ndcg"]), _color_for(ir_agg["ndcg"]) if evaluated > 0 else COLORS["text_muted"],
     "Ranking quality of top-k results"),
    ("Avg Response", f"{avg_resp:.2f}s", COLORS["accent"], "End-to-end response latency"),
    ("Avg Retrieval", f"{avg_ret:.2f}s", COLORS["accent"], "Retriever latency"),
    ("Indexed Docs", str(total_docs_indexed), COLORS["accent"], "Total PDFs indexed so far"),
]

core_cols = st.columns(5)
for col, (label, value, color, desc) in zip(core_cols, core_metrics):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color:{color};">{value}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-sub">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Trend chart ───────────────────────────────────────────────────────────────
if per_q:
    st.markdown('<div class="section-title">📈 Metrics Trend per Query</div>', unsafe_allow_html=True)

    df_trend = pd.DataFrame(per_q)
    df_scored = df_trend[df_trend["precision"].notna()].copy()
    df_scored = df_scored.reset_index(drop=True)
    df_scored["#"] = df_scored.index + 1

    if not df_scored.empty:
        fig_trend = go.Figure()
        metric_series = [
            ("nDCG@k",       "ndcg",       COLORS["chart4"]),
            ("HitRate@k",    "hit_rate",   COLORS["chart5"]),
        ]
        for name, col, color in metric_series:
            fig_trend.add_trace(go.Scatter(
                x=df_scored["#"].tolist(),
                y=df_scored[col].tolist(),
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
                hovertemplate=f"<b>{name}</b><br>Query #%{{x}}<br>Score: %{{y:.4f}}<extra></extra>",
            ))

        fig_trend.update_layout(
            paper_bgcolor=COLORS["bg_card"],
            plot_bgcolor=COLORS["bg_secondary"],
            font=dict(family="Inter, sans-serif", color=COLORS["text_primary"]),
            margin=dict(l=16, r=16, t=32, b=16),
            height=320,
            xaxis=dict(
                title="Query #",
                gridcolor=COLORS["border"],
                linecolor=COLORS["border"],
                autorange=True,
            ),
            yaxis=dict(
                title="Score (0 – 1)",
                range=[0, 1.05],
                gridcolor=COLORS["border"],
                linecolor=COLORS["border"],
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor=COLORS["bg_card"], bordercolor=COLORS["border"],
            ),
        )
        st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No per-query score data available yet.")

# ── Response time charts ──────────────────────────────────────────────────────
st.markdown('<div class="section-title">⏱️ Response Time Trend</div>', unsafe_allow_html=True)

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
st.markdown('<div class="section-title">📊 Aggregate Statistics</div>', unsafe_allow_html=True)

agg_cols = st.columns(4)
agg_stats = [
    ("Total Queries",      str(total_q),          "queries"),
    ("Avg Response Time",  f"{avg_resp:.2f}s",     "total"),
    ("Avg Retrieval Time", f"{avg_ret:.2f}s",      "retrieval"),
    ("Avg Docs Retrieved", f"{avg_docs:.1f}",      "per query"),
]
for col, (label, val, sub) in zip(agg_cols, agg_stats):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color:{COLORS['accent']};">{val}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Per-query table ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🗂️ Query History (Core)</div>', unsafe_allow_html=True)

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

    adv_metric_defs = [
        ("Precision@k", ir_agg["precision"], "Fraction of top-k docs that are relevant"),
        ("Recall@k", ir_agg["recall"], "Fraction of relevant docs found in top-k"),
        ("MRR", ir_agg["mrr"], "How early the first relevant doc appears"),
        ("Avg Generation", avg_gen, "Average answer-generation latency (s)"),
    ]

    adv_cols = st.columns(4)
    for col, (label, val, desc) in zip(adv_cols, adv_metric_defs):
        if label == "Avg Generation":
            display_value = f"{val:.2f}s"
            color = COLORS["accent"]
        else:
            display_value = _fmt(val)
            color = _color_for(val) if evaluated > 0 else COLORS["text_muted"]

        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color:{color};">{display_value}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-sub">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📚 Metric Definitions</div>', unsafe_allow_html=True)
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
            st.markdown(f"""
            <div class="kpi-card" style="text-align:left; min-height:160px;">
                <div style="font-size:13px; font-weight:600; color:{color}; margin-bottom:8px;">{title}</div>
                <div style="font-size:12px; color:{COLORS['text_secondary']}; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(formula)
