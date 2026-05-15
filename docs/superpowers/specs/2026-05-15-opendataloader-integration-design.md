# Design Doc: OpenDataLoader PDF Integration for High-Fidelity RAG

## Overview
Replace the current `PyMuPDFLoader` with `OpenDataLoader PDF` to improve table extraction accuracy, especially for financial and corporate documents. The pipeline will transition from plain-text extraction to Markdown-based, structure-aware extraction, enabling the RAG system to better understand tables and document hierarchies.

## Goals
1. **Improve Table Accuracy**: Correctly extract and format complex tables into Markdown.
2. **Structure-Aware Chunking**: Use Markdown headers to maintain context during retrieval.
3. **HF Spaces Compatibility**: Ensure the pipeline runs on Hugging Face Spaces by installing Java 11+ via Docker.
4. **Local-First Processing**: Keep all document processing local and private (no external API dependencies for parsing).

## Infrastructure Changes

### Docker & System Dependencies
- **Dockerfile**:
  - Add `openjdk-17-jre-headless` to the system package installation step.
  - Ensure `JAVA_HOME` is set if necessary (though usually handled by standard debian/ubuntu packages).
- **requirements.txt**:
  - Add `opendataloader-pdf`
  - Add `langchain-opendataloader-pdf`

## Implementation Details

### 1. Document Loading (`backend/app/index_data.py`)
- Replace `PyMuPDFLoader` with `OpenDataLoaderPDFLoader`.
- Configure loader with `format="markdown"` and `mode="local"`.
- Implement basic error handling for Java availability.

### 2. Structure-Aware Chunking
- Use `MarkdownHeaderTextSplitter` to split documents by headers (`#`, `##`, `###`).
- Follow up with `RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN)` to handle large sections while respecting Markdown syntax (avoiding breaking tables mid-row).
- **Metadata**: Preserve `source_file`, `page_number`, and inject header context into each chunk's metadata.

### 3. Vector Store Upsert
- No changes required to Qdrant logic, but since Markdown is more verbose than plain text, we might need to monitor the average chunk size.

## Data Flow
1. **Upload**: User uploads a PDF via the frontend.
2. **Load**: `OpenDataLoaderPDFLoader` (via Java engine) parses the PDF into a structured Markdown string.
3. **Split**: 
   - Step A: Split by Markdown headers (injecting header path into metadata).
   - Step B: Split remaining large blocks by characters (Markdown-optimized).
4. **Embed**: Chunks are embedded using `FastEmbed`.
5. **Upsert**: Documents (text + metadata) are stored in Qdrant.

## Verification Plan

### Automated Tests
- Create a test script `backend/tests/test_opendataloader_integration.py`.
- Verify that a sample financial PDF with tables is correctly converted to Markdown tables.
- Verify that headers are correctly captured in metadata.
- Test Java dependency detection.

### Manual Verification
- Index a real-world financial document (e.g., an annual report) and check the "Source Documents" in the UI to see if tables look like tables.
- Deploy a temporary branch to a Hugging Face Space to confirm the Docker/Java setup works.

## Risks & Mitigations
- **Java Requirement**: Increases Docker image size. *Mitigation: Use `jre-headless` to minimize footprint.*
- **Parsing Speed**: Slightly slower than PyMuPDF. *Mitigation: OpenDataLoader's heuristic engine is still very fast (>60 pages/sec).*
- **Broken Tables**: Very wide tables might still be hard for LLMs to read. *Mitigation: Using Markdown format is the best known way to present tables to LLMs.*
