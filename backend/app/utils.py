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
            "page": metadata.get("page_number") or metadata.get("page", 0),
            "header": metadata.get("Header 1") or metadata.get("Header 2") or metadata.get("Header 3") or "",
            "score": metadata.get("score", 0.0)
        })

    sources_str = "\n".join(f"- {filename}" for filename in sorted(source_files))
    return context, sources_str, sources_json


class ThinkingParser:
    """Stateful parser for streaming RAG responses with <thinking> tags."""

    def __init__(self):
        self.buffer = ""
        self.inside_thinking = False
        self.thinking_done = False

    def feed(self, chunk: str) -> Iterable[Tuple[str, str]]:
        """Feed a chunk of text and yield (type, content) pairs."""
        self.buffer += chunk

        while self.buffer:
            if not self.inside_thinking and not self.thinking_done:
                start_tag = "<thinking>"
                if start_tag in self.buffer:
                    idx = self.buffer.find(start_tag)
                    pre_content = self.buffer[:idx]
                    if pre_content:
                        yield "token", pre_content
                    self.inside_thinking = True
                    self.buffer = self.buffer[idx + len(start_tag):]
                    continue
                
                # If we have a '<' but not the full start_tag, check if it COULD be the start_tag
                if "<" in self.buffer:
                    idx = self.buffer.find("<")
                    # If there's text before '<', yield it
                    if idx > 0:
                        yield "token", self.buffer[:idx]
                        self.buffer = self.buffer[idx:]
                        continue
                    
                    # Buffer starts with '<'. Is it a potential start_tag?
                    if len(self.buffer) < len(start_tag):
                        # Could be start_tag, wait for more data
                        if start_tag.startswith(self.buffer):
                            break
                        else:
                            # Not a start_tag (e.g. "<b"), yield the "<" and continue
                            yield "token", self.buffer[0]
                            self.buffer = self.buffer[1:]
                            continue
                    else:
                        # Buffer is long enough but doesn't match start_tag
                        yield "token", self.buffer[0]
                        self.buffer = self.buffer[1:]
                        continue
                else:
                    # No '<' at all, yield everything
                    yield "token", self.buffer
                    self.buffer = ""
                    break

            elif self.inside_thinking:
                end_tag = "</thinking>"
                if end_tag in self.buffer:
                    idx = self.buffer.find(end_tag)
                    thought_content = self.buffer[:idx]
                    if thought_content:
                        yield "thinking", thought_content
                    self.inside_thinking = False
                    self.thinking_done = True
                    self.buffer = self.buffer[idx + len(end_tag):]
                    continue
                
                if "</" in self.buffer:
                    idx = self.buffer.find("</")
                    if idx > 0:
                        yield "thinking", self.buffer[:idx]
                        self.buffer = self.buffer[idx:]
                        continue
                    
                    if len(self.buffer) < len(end_tag):
                        if end_tag.startswith(self.buffer):
                            break
                        else:
                            yield "thinking", self.buffer[0]
                            self.buffer = self.buffer[1:]
                            continue
                    else:
                        yield "thinking", self.buffer[0]
                        self.buffer = self.buffer[1:]
                        continue
                else:
                    yield "thinking", self.buffer
                    self.buffer = ""
                    break
            
            else:
                # After thinking is done, everything is a token
                yield "token", self.buffer
                self.buffer = ""
                break

    def flush(self) -> Iterable[Tuple[str, str]]:
        """Yield any remaining content in the buffer."""
        if not self.buffer:
            return
        
        if self.inside_thinking:
            yield "thinking", self.buffer
        else:
            yield "token", self.buffer
        self.buffer = ""