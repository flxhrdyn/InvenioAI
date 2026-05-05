"""Small helpers shared across the RAG pipeline."""

from __future__ import annotations

from typing import Any, Iterable, Tuple


def format_docs(docs: Iterable[Any]) -> Tuple[str, str]:
    """Build the context block and a readable sources list.

    Args:
        docs: Iterable of LangChain `Document`-like objects.

    Returns:
        A tuple `(context, sources)` where `context` is chunk text joined with
        blank lines and `sources` is a bullet list of unique filenames derived
        from `metadata['source']`.
    """

    docs_list = list(docs)
    context = "\n\n".join(doc.page_content for doc in docs_list)

    # Keep sources stable and readable: filename only, unique, sorted.
    source_files: set[str] = set()
    for doc in docs_list:
        metadata = getattr(doc, "metadata", None) or {}
        source_path = metadata.get("source", "unknown")
        if not isinstance(source_path, str):
            source_path = "unknown"
        filename = source_path.split("\\")[-1].split("/")[-1]
        source_files.add(filename)

    sources = "\n".join(f"- {filename}" for filename in sorted(source_files))
    return context, sources