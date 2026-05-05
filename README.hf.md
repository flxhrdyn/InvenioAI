---
title: InvenioAI
emoji: "🧠"
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# InvenioAI (Docker Space)

This Space runs InvenioAI as a single Docker container:

- FastAPI backend (internal): `http://127.0.0.1:8000`
- Streamlit UI (public): `http://0.0.0.0:${PORT}` (Hugging Face usually sets `PORT=7860`)

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