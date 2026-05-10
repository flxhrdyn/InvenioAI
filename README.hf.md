---
title: InvenioAI - RAG Dashboard
emoji: 🚀
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🧠 InvenioAI — Advanced RAG & Analytics
**Production-Ready Document Intelligence with Llama 3.1, Hybrid Search, and RAG Fusion.**

InvenioAI is a high-performance Document Q&A system designed to transform static PDFs into actionable knowledge. It implements an advanced RAG architecture with a heavy focus on retrieval precision and observability.

### 🚀 Core Capabilities
- **Advanced RAG Architecture**: Multi-query generation (RAG Fusion), Hybrid Search (Dense + Lexical), and **FlashRank Reranking** (ms-marco-MultiBERT-L-12).
- **Semantic Caching**: Dual-layer caching strategy with a high-performance semantic lookup (Cosine Similarity > 0.90) to eliminate redundant LLM API calls and provide near-instant responses.
- **Chain-of-Thought (CoT)**: Structured 4-step reasoning protocol for high-fidelity response generation.
- **Production-Ready UX**: Premium adaptive UI with 'Outfit' typography and a dedicated **Retrieval Metrics Dashboard** (nDCG, HitRate).
- **Fast & Scalable**: Powered by Groq (Llama 3.1) and Qdrant for sub-second inference and robust indexing.

## 🛠️ Infrastructure (Hugging Face Docker)
This Space runs as a single Docker container:
- **FastAPI Backend**: Internal orchestration and RAG logic.
- **Streamlit Frontend**: Public interface on port `7860`.

## Required secrets / variables (Space Settings)

Required:

- Secret: `GROQ_API_KEY`

Recommended:

- Secret: `QDRANT_API_KEY` (when using Qdrant Cloud)
- Variable: `QDRANT_URL` (when using Qdrant Cloud)
- Variable: `INVENIOAI_DELETE_UPLOADED_PDFS=1`
- Variable: `INVENIOAI_UPLOAD_TIMEOUT_SECONDS=600` (optional, for large PDFs/cold starts)
- Variable: `QDRANT_PREFER_GRPC=0`
- Variable: `STREAMLIT_ENABLE_CORS` (optional override)
- Variable: `STREAMLIT_ENABLE_XSRF_PROTECTION` (optional override)

## Notes

- This Space is optimized for a monorepo structure. For production deployments with Docker Compose, see `docker-compose.yml`.
- First build can take a while because ML dependencies (FastEmbed) are downloaded during first run.
- Ensure `GROQ_API_KEY` is set in Space Secrets.
