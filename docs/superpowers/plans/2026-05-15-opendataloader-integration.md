# OpenDataLoader PDF Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `PyMuPDFLoader` with `OpenDataLoaderPDFLoader` to improve table extraction and implement structure-aware chunking for RAG.

**Architecture:** Use `OpenDataLoaderPDF` as the local-first Java-based parsing engine. Extract documents as Markdown, then split them using `MarkdownHeaderTextSplitter` (for hierarchy) followed by `RecursiveCharacterTextSplitter` (for size) to preserve table integrity and context.

**Tech Stack:** `opendataloader-pdf`, `langchain-opendataloader-pdf`, `langchain`, `openjdk-17-jre-headless`.

---

### Task 1: Infrastructure & Dependencies

**Files:**
- Modify: `Dockerfile`
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Update requirements.txt**

Add the new libraries to `backend/requirements.txt`.

```text
# ... existing ...
opendataloader-pdf>=1.0.0
langchain-opendataloader-pdf>=0.1.0
```

- [ ] **Step 2: Update Dockerfile**

Install Java 17 JRE in the final stage of the `Dockerfile`.

```dockerfile
# ... existing in final stage ...
FROM python:3.12-slim-bookworm

# Install Java 17 for OpenDataLoader
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*
# ... rest of file ...
```

- [ ] **Step 3: Commit**

```bash
git add backend/requirements.txt Dockerfile
git commit -m "infra: add java 17 and opendataloader dependencies"
```

---

### Task 2: Implement OpenDataLoader Integration

**Files:**
- Modify: `backend/app/index_data.py`

- [ ] **Step 1: Update imports and loader logic**

Replace `PyMuPDFLoader` with `OpenDataLoaderPDFLoader` in `backend/app/index_data.py`.

```python
# backend/app/index_data.py

from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language
)

def process_pdf_documents(file_path: str) -> list[Document]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # Use OpenDataLoader for high-fidelity Markdown
    loader = OpenDataLoaderPDFLoader(
        file_path=file_path,
        format="markdown"
    )
    docs = loader.load()
    
    # Combined splitting strategy
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    final_chunks = []
    for doc in docs:
        # 1. Split by headers
        header_splits = markdown_splitter.split_text(doc.page_content)
        
        # 2. Recursive split for size
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True
        )
        
        for split in header_splits:
            chunks = text_splitter.create_documents([split.page_content], metadatas=[split.metadata])
            for chunk in chunks:
                # Merge original metadata (page, source) with header metadata
                chunk.metadata.update(doc.metadata)
                chunk.metadata["source_file"] = path.name
                if "page" in chunk.metadata:
                    chunk.metadata["page_number"] = chunk.metadata["page"] + 1
                final_chunks.append(chunk)
                
    return final_chunks
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/index_data.py
git commit -m "feat: implement opendataloader with markdown splitting"
```

---

### Task 3: Verification & Tests

**Files:**
- Create: `backend/tests/test_opendataloader_integration.py`

- [ ] **Step 1: Write integration test**

Create a test to verify the new loader works.

```python
# backend/tests/test_opendataloader_integration.py
import pytest
from pathlib import Path
from backend.app.index_data import process_pdf_documents

def test_process_pdf_with_opendataloader():
    # Note: Requires java installed in the environment to pass
    sample_pdf = Path("backend/tests/test_pdf.pdf")
    if not sample_pdf.exists():
        pytest.skip("Sample PDF not found")
        
    chunks = process_pdf_documents(str(sample_pdf))
    
    assert len(chunks) > 0
    # Check if table structure is preserved (look for |)
    has_table = any("|" in chunk.page_content for chunk in chunks)
    assert has_table, "Markdown table structure not found in chunks"
    assert "page_number" in chunks[0].metadata
```

- [ ] **Step 2: Run tests**

```bash
pytest backend/tests/test_opendataloader_integration.py -v
```

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_opendataloader_integration.py
git commit -m "test: add opendataloader integration test"
```

---

### Task 4: UI Verification (Manual)

- [ ] **Step 1: Test with real financial document**

Upload a financial PDF through the UI and verify in the logs or "Source Documents" view that tables are rendered as Markdown.
