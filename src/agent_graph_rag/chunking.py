"""Document loading and chunking.

A document is a Markdown file under ``data/``. Pages are separated by a
``--- Page N ---`` marker line; each page becomes one chunk, so provenance
can point at an exact (document, page) pair. Chunks keep their source text
so answers can cite the verbatim span.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PAGE_MARKER = re.compile(r"^---\s*Page\s+(\d+)\s*---\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str  # e.g. "lab-note-01-synthesis:p1"
    page: int
    text: str

    @property
    def provenance_kwargs(self) -> dict:
        return {"doc_id": self.doc_id, "chunk_id": self.chunk_id, "page": self.page}


@dataclass(frozen=True)
class Document:
    doc_id: str
    path: Path
    text: str

    def chunks(self) -> list[Chunk]:
        return chunk_document(self.doc_id, self.text)


def chunk_document(doc_id: str, text: str) -> list[Chunk]:
    """Split text on page markers; text before the first marker belongs to
    page 1 (merged into the page-1 body when one exists)."""
    parts = PAGE_MARKER.split(text)
    # split() with a capture group: [pre, "1", body, "2", body, ...]
    segments: list[tuple[int, str]] = []
    for i in range(1, len(parts), 2):
        page = int(parts[i])
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if body:
            segments.append((page, body))
    pre = parts[0].strip()
    if pre:
        if segments and segments[0][0] == 1:
            segments[0] = (1, pre + "\n" + segments[0][1])
        else:
            segments.insert(0, (1, pre))
    if not segments and pre:
        segments.append((1, pre))
    return [Chunk(doc_id, f"{doc_id}:p{page}", page, body)
            for page, body in segments]


def load_documents(data_dir: str | Path) -> list[Document]:
    """Load every ``*.md`` file in ``data_dir``, sorted by filename."""
    data_dir = Path(data_dir)
    docs = []
    for path in sorted(data_dir.glob("*.md")):
        docs.append(Document(doc_id=path.stem, path=path, text=path.read_text()))
    return docs


def load_chunks(data_dir: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in load_documents(data_dir):
        chunks.extend(doc.chunks())
    return chunks
