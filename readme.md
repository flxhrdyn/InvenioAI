<div align="center">

  # 🧠 InvenioAI — Advanced RAG for Document Q&A
  **Hybrid Search, RAG Fusion, and Chain-of-Thought (CoT) Reasoning.**
  
  [![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
  [![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com/)
  [![Qdrant](https://img.shields.io/badge/Qdrant-FF4B4B?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech/)
  [![Groq](https://img.shields.io/badge/Groq-Llama_3.1-f3a536?style=for-the-badge&logo=openai&logoColor=white)](https://groq.com/)
  [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
</div>

---

## Overview

In the era of information density, extracting precise answers from large PDF collections is critical. **InvenioAI** is a high-performance **Advanced RAG system** that implements a state-of-the-art **Hybrid architecture**.

It transforms static PDF documents into a searchable, intelligent knowledge base, allowing users to ask complex questions and receive answers grounded in multi-stage retrieved context with verifiable source citations.

## Live Demo

- **Hugging Face Space**: [https://felixhrdyn-invenioai.hf.space](https://felixhrdyn-invenioai.hf.space)

## Technical Features

- **Hybrid Search**: Combines dense semantic retrieval (MMR) with lexical BM25 search, fused via weighted Reciprocal Rank Fusion (RRF).
- **RAG Fusion**: Implements Multi-Query generation to capture diverse user intents and improve retrieval coverage.
- **Advanced Reranking**: Utilizes Cross-Encoder models (`ms-marco-MultiBERT-L-12`) via FlashRank to re-evaluate top candidates, ensuring the most relevant context is provided to the LLM.
- **Chain-of-Thought (CoT) Reasoning**: Implements a 4-step structured reasoning protocol (Query Deconstruction, Filtering, Synthesis, Strategy) to ensure grounded and logical answers.
- **Async Job Orchestration**: Background indexing and query execution with real-time status polling for a smooth user experience.
- **Deep Analytics Dashboard**: Built-in metrics tracking for retrieval accuracy (nDCG, HitRate), latency, and API usage.
- **Minimalist UI/UX**: Centered branding with 'Outfit' typography, glassmorphism aesthetics, and a streamlined Knowledge Base management interface.
- **Cloud-Ready Architecture**: Ships with an all-in-one Docker configuration optimized for Hugging Face Spaces and Azure Container Apps.

## Technology Stack

### Backend
- **Framework**: FastAPI
- **RAG Engine**: LangChain
- **LLM**: Llama 3.3 70B & Llama 3.1 8B (Groq Cloud)
- **Reasoning**: Chain-of-Thought (CoT) structured 4-step protocol
- **Embedding Model**: paraphrase-multilingual-MiniLM-L12-v2 (Local)
- **Reranker**: FlashRank (ms-marco-MultiBERT-L-12 Cross-Encoder)
- **Search**: BM25 (Lexical) + Qdrant (Dense) + RAG Fusion (Multi-Query)

### Frontend
- **Framework**: Streamlit
- **Visualization**: Plotly, Pandas
- **Styling**: Vanilla CSS (Custom Design System)
- **Icons**: Lucide (SVG)

### Infrastructure
- **Vector Database**: Qdrant (Local / Server / Cloud)
- **Deployment**: Docker, GitHub Actions (CI/CD)
- **Environment**: Python 3.10+

## System Architecture

```mermaid
graph TD
    subgraph Data_Layer [Ingestion Layer]
        PDF[PDF Documents] -->|Upload| API[FastAPI Backend]
        API -->|Chunking| Split[Text Splitter]
    end
    
    subgraph Intelligence_Layer [Processing & RAG]
        Split -->|Dense| QDR[Qdrant Vector DB]
        Split -->|Lexical| BM25[BM25 Index]
        
        API -->|Query Rewriting| Rewriter[Query Rewriter]
        Rewriter -->|Multi-Query| RAG[Hybrid Retriever]
        RAG -->|RRF Fusion| Fuse[RAG Fusion]
        Fuse -->|Reranking| Rerank[Cross-Encoder]
        Rerank -->|Context| LLM[Groq (Llama 3.1) LLM]
    end
    
    subgraph Presentation_Layer [UI & Analytics]
        UI[Streamlit Dashboard] -->|REST API| API
        LLM -->|Answer| UI
        API -->|Log| Metrics[Local Metrics Store]
        Metrics -->|Visualize| Dashboard[Analytics Page]
    end
```

---

## Performance & Limits

InvenioAI is optimized for speed and retrieval precision while maintaining low operational costs.

### Core Metrics & Operational Limits
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Retrieval Mode** | **Hybrid** | Dense (MMR) + Lexical (BM25) |
| **Rerank Top-K** | **5 Docs** | Optimized context window for LLM |
| **Avg. Response** | **~34s** | Total end-to-end latency (RAG Fusion + Reranking) |
| **Avg. Retrieval** | **~12s** | Multi-query hybrid search & RRF fusion time |

---

## Deployment Guide

### Prerequisites
*   Python 3.10+
*   Google Groq (Llama 3.1) API Key
*   Qdrant Instance (Optional, defaults to local storage)

### Execution Procedures

**Step 1: Environment Setup**
```bash
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
```

**Step 2: Run Application**
```bash
# Terminal 1: Backend API
uvicorn app.main:app --reload

# Terminal 2: Streamlit UI
streamlit run frontend/streamlit_app.py
```

**Step 3: Docker (Production)**
```bash
docker build -t invenioai .
docker run -p 7860:7860 invenioai
```

---

## Configuration

The application is configured via `.env`. Key variables include:
- `GROQ_API_KEY`: Required for LLM and Query Rewriting.
- `QDRANT_URL`: Optional server URL (defaults to local `./qdrant_storage`).
- `INVENIOAI_ENABLE_HYBRID_SEARCH`: Toggle dense+lexical mode (Default: `1`).
- `INVENIOAI_DELETE_UPLOADED_PDFS`: Clean up storage after indexing (Default: `0`).

---

## Author

**Felix Hardyan**
*   [GitHub](https://github.com/flxhrdyn)
*   [Hugging Face](https://huggingface.co/felixhrdyn)
