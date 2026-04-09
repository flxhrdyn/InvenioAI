---
title: DocuMind
emoji: "🧠"
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# DocuMind (Docker Space)

This Space runs DocuMind as a single Docker container:

- FastAPI backend (internal): `http://127.0.0.1:8000`
- Streamlit UI (public): `http://0.0.0.0:${PORT}` (Hugging Face usually sets `PORT=7860`)

## Required secrets / variables (Space Settings)

Required:

- Secret: `GEMINI_API_KEY`

Recommended:

- Secret: `QDRANT_API_KEY` (when using Qdrant Cloud)
- Variable: `QDRANT_URL` (when using Qdrant Cloud)
- Variable: `DOCUMIND_DELETE_UPLOADED_PDFS=1`
- Variable: `QDRANT_PREFER_GRPC=0`

## Notes

- This Space is self-contained: Streamlit calls the backend at `http://127.0.0.1:8000` inside the same container.
- First build can take a while because ML dependencies may be large.