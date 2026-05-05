"""Small helpers shared across the RAG pipeline."""

from __future__ import annotations

from typing import Any, Iterable, Tuple, List, Dict


def format_docs(docs: Iterable[Any]) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Build the context block and a readable sources list with structured metadata.

    Args:
        docs: Iterable of LangChain `Document`-like objects.

    Returns:
        A tuple `(context, sources_str, sources_json)`
    """

    docs_list = list(docs)
    context = "\n\n".join(doc.page_content for doc in docs_list)

    # Structured metadata for the frontend
    sources_json = []
    source_files: set[str] = set()

    for doc in docs_list:
        metadata = getattr(doc, "metadata", None) or {}
        source_path = metadata.get("source", metadata.get("source_file", "unknown"))
        if not isinstance(source_path, str):
            source_path = "unknown"
        filename = source_path.split("\\")[-1].split("/")[-1]
        source_files.add(filename)
        
        sources_json.append({
            "file": filename,
            "text": doc.page_content,
            "score": metadata.get("score", 0.0)
        })

    sources_str = "\n".join(f"- {filename}" for filename in sorted(source_files))
    return context, sources_str, sources_json