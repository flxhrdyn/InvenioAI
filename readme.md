# 🧠 DocuMind — Document Q&A (RAG) + Analytics

**Core Stack**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-teal?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker)

**AI / RAG**

![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-6C63FF)
![Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)

**Delivery**

[![CI](https://github.com/flxhrdyn/Documind/actions/workflows/ci.yml/badge.svg)](https://github.com/flxhrdyn/Documind/actions/workflows/ci.yml)
[![CD](https://github.com/flxhrdyn/Documind/actions/workflows/sync-to-hf-space.yml/badge.svg)](https://github.com/flxhrdyn/Documind/actions/workflows/sync-to-hf-space.yml)
![HuggingFace%20Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-FFD21E?logo=huggingface)
![GitHub%20Actions](https://img.shields.io/badge/GitHub%20Actions-Automation-2088FF?logo=githubactions)

**DocuMind** is an MVP-ready document question-answering system built with **FastAPI** (backend) and **Streamlit** (frontend). It runs a **RAG pipeline** (query rewriting → retrieval → reranking → answer generation) backed by **Qdrant**.

---

## 🌟 Features

- **📄 PDF indexing**: Upload and index multiple PDF documents
- **🔍 RAG pipeline**: Query rewriting + semantic retrieval + reranking
- **⚡ Vector search**: Qdrant vector store (local or server/cloud)
- **📊 Analytics**: Local metrics file + a Streamlit dashboard page
- **🔐 Config via env**: `.env` / `.env.example` for secrets and runtime flags

---

## 🔗 Live Demo

This repo does not ship public demo links by default. Local endpoints:

- Frontend (Streamlit): `http://localhost:8501`
- Backend (FastAPI): `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

For hosted deployments, this repo provides:

- A canonical all-in-one Dockerfile for Hugging Face Spaces (FastAPI + Streamlit)
- An optional backend-only Dockerfile for local/dev use
- A GitHub Actions workflow to deploy the **full Docker Space** (FastAPI + Streamlit)

---

## 🎯 Problem Statement

When you have a set of PDF documents, it’s tedious to manually search for answers. DocuMind turns PDFs into a searchable knowledge base and lets you ask questions with answers grounded in retrieved context, while tracking basic performance metrics.

---

## 🏗️ RAG Pipeline & Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  Frontend   │────▶│   FastAPI    │─────▶│     LLM     │
│ (Streamlit) │      │   Backend    │      │   (Gemini)  │
└─────────────┘      └──────────────┘      └─────────────┘
       │                    │
       │                    ▼
       │             ┌──────────────┐
       │             │  RAG Pipeline│
       │             └──────────────┘
       │                    │
       ▼                    ▼
┌─────────────┐      ┌──────────────┐
│  Metrics    │      │   Qdrant     │
│ (local JSON)│      │ Vector Store │
└─────────────┘      └──────────────┘
```

Pipeline (high-level):

1. Rewrite question into a standalone query (LLM)
2. Retrieve relevant chunks from Qdrant (MultiQueryRetriever)
3. Rerank top chunks (CrossEncoder)
4. Generate answer constrained to retrieved context (LLM)

---

## 🛠️ Tech Stack

| Layer | Tech |
|------:|------|
| Backend API | FastAPI + Uvicorn |
| Frontend UI | Streamlit |
| Vector DB | Qdrant (local or server/cloud) |
| RAG | LangChain + langchain-qdrant |
| LLM | Google Gemini (via `langchain-google-genai`) |
| Embeddings | Sentence Transformers |
| Reranker | CrossEncoder (sentence-transformers) |
| Testing | pytest |
| CI | GitHub Actions |
| CD / Deployment | Hugging Face Spaces (Docker) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Google Gemini API key

### Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set at minimum:

```bash
GEMINI_API_KEY=...
```

### Run backend

```bash
uvicorn app.main:app --reload
```

### Run frontend

```bash
streamlit run frontend/streamlit_app.py
```

---

## 📖 Usage

1. Upload a PDF from the sidebar
2. Click **Index Document**
3. Ask questions in the chat input
4. Open the **Dashboard** page to see metrics

---

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest
```

---

## 📊 Dashboard

The Streamlit dashboard page shows:

- Total queries
- Total documents indexed
- Average response time
- Per-query history (timings + retrieval scores when available)

Note: metrics are written to a local `metrics.json` file and are not shared across multiple instances.

---

## 🔧 Configuration

Common settings (see `.env.example`):

```bash
# Required
GEMINI_API_KEY=...

# Optional: Qdrant server/cloud (recommended for multi-process)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_PREFER_GRPC=0

# Optional: Streamlit -> backend URL
DOCUMIND_API_BASE_URL=http://localhost:8000

# Optional: delete PDFs after indexing
DOCUMIND_DELETE_UPLOADED_PDFS=0

# Optional: upload/indexing timeout for Streamlit -> backend (seconds)
# Useful for large PDFs and cold starts
DOCUMIND_UPLOAD_TIMEOUT_SECONDS=600

# Optional: Hugging Face Hub timeouts
HF_HUB_CONNECT_TIMEOUT=30
HF_HUB_READ_TIMEOUT=120

# Optional: Streamlit proxy/security flags
# Local/dev default: true/true
# HF Spaces default via start.sh: false/false (unless explicitly set)
# STREAMLIT_ENABLE_CORS=true
# STREAMLIT_ENABLE_XSRF_PROTECTION=true
```

### Qdrant local storage lock (important)

By default the project can use **local Qdrant storage** at `qdrant_storage/`. Local mode can be accessed by only **one client instance per process**, and can fail under multi-process/concurrent usage.

If you see:

`RuntimeError: Storage folder ... is already accessed by another instance of Qdrant client`

use **Qdrant server/cloud mode** by setting `QDRANT_URL`.

---

## 🐳 Deployment

### Docker (all-in-one)

```bash
docker build -t documind .
docker run -p 7860:7860 documind
```

Then open `http://localhost:7860`.

### Docker (backend-only, optional)

```bash
docker build -t documind-api -f Dockerfile.api .
docker run -p 8000:8000 documind-api
```

### Hugging Face Spaces (Docker) via CI/CD

This repo includes a GitHub Actions workflow that syncs a clean snapshot of the repo to a Hugging Face **Docker** Space on every push to `main`.

The Space runs a single container that starts:

- FastAPI backend (internal): `127.0.0.1:8000`
- Streamlit UI (public): `0.0.0.0:7860` (Hugging Face provides `PORT=7860`)

1) Create a **Docker** Space on Hugging Face.

2) Configure runtime values in the Space.

Space path: **Settings -> Variables and secrets**

Secrets (recommended):

- `GEMINI_API_KEY` (required)
- `QDRANT_API_KEY` (if using Qdrant Cloud)

Variables (recommended):

- `QDRANT_URL` (if using Qdrant Cloud)
- `DOCUMIND_DELETE_UPLOADED_PDFS=1` (optional, useful on ephemeral disks)
- `DOCUMIND_UPLOAD_TIMEOUT_SECONDS=600` (optional, for large PDFs/cold starts)
- `QDRANT_PREFER_GRPC=0` (optional; HTTP/REST is often more reliable)
- `STREAMLIT_ENABLE_CORS` (optional override)
- `STREAMLIT_ENABLE_XSRF_PROTECTION` (optional override)

Streamlit security defaults are environment-aware in `start.sh`:

- Local/dev: `enableCORS=true`, `enableXsrfProtection=true`
- HF Spaces: defaults to `false/false` for proxy compatibility
- Explicit env vars always override defaults

3) Configure deployment credentials in GitHub Actions.

GitHub path: **Settings -> Secrets and variables -> Actions**

Repository secrets:

- `HF_TOKEN`: Hugging Face token with **write** permission
- `HF_SPACE_ID`: `username/space-name`
- `HF_SPACE_BRANCH`: `main` (optional)

4) Push to `main` (or run the workflow manually from the Actions tab).

Workflow file: `.github/workflows/sync-to-hf-space.yml`

Space card source file in this repo: `README.hf.md`

#### Secrets vs Variables quick rule

- Use **Secrets** for credentials/tokens/API keys.
- Use **Variables** for non-sensitive config flags and URLs.

#### Common Spaces form mistake (important)

When adding values in Hugging Face Spaces:

- The **Key** field must contain only the variable name, for example `QDRANT_API_KEY`.
- The **Value** field must contain only the value.
- Do **not** paste full `.env` lines like `QDRANT_API_KEY="..."` into Key.
- Avoid extra spaces around `=` in `.env` files.

---

## 📁 Project Structure

```
documind-test/
├── app/                     # FastAPI backend + RAG pipeline
├── frontend/                # Streamlit UI + dashboard page
├── tests/                   # Unit tests
├── uploaded_docs/           # Local PDF storage (optional)
├── qdrant_storage/          # Local Qdrant storage (optional)
├── .env.example
├── Dockerfile               # Canonical all-in-one image (HF/local)
├── Dockerfile.api           # Optional backend-only image
├── start.sh                 # Entrypoint for all-in-one container
├── README.hf.md             # Space card template used by CI sync
├── requirements.txt
└── readme.md
```

---

## ⚠️ Limitations (current MVP)

- `/query/jobs` state is stored in memory (lost on restart)
- Dashboard metrics are stored in a local `metrics.json` file (not shared across instances)
- No authentication / rate limiting built-in
