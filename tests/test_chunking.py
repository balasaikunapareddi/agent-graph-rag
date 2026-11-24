"""Chunking tests: page markers become chunks with exact provenance."""

from pathlib import Path

import pytest

from agent_graph_rag.chunking import (
    chunk_document,
    load_chunks,
    load_documents,
)

DATA = Path(__file__).parent.parent / "src" / "agent_graph_rag" / "data"


def test_page_markers_split_into_pages():
    chunks = chunk_document("doc", "--- Page 1 ---\nbody one\n--- Page 2 ---\nbody two")
    assert [(c.page, c.chunk_id) for c in chunks] == [
        (1, "doc:p1"), (2, "doc:p2")]
    assert "body two" in chunks[1].text


def test_text_before_first_marker_is_page_1():
    chunks = chunk_document("doc", "header line\n--- Page 2 ---\nsecond")
    assert chunks[0].page == 1 and "header line" in chunks[0].text
    assert chunks[1].page == 2


def test_no_markers_is_a_single_chunk():
    chunks = chunk_document("doc", "just some text")
    assert len(chunks) == 1 and chunks[0].page == 1


def test_shipped_docs_load():
    docs = load_documents(DATA)
    assert len(docs) == 5
    assert docs[0].doc_id == "lab-note-01-synthesis"


def test_shipped_docs_produce_ten_chunks():
    chunks = load_chunks(DATA)
    assert len(chunks) == 10
    pages = sorted({(c.doc_id, c.page) for c in chunks})
    assert len(pages) == 10


def test_chunk_provenance_kwargs():
    chunk = load_chunks(DATA)[0]
    kwargs = chunk.provenance_kwargs
    assert kwargs["doc_id"] == chunk.doc_id
    assert kwargs["chunk_id"] == chunk.chunk_id
    assert kwargs["page"] == chunk.page


def test_page_zero_marker_rejected():
    # Page 0 would stamp provenance Pydantic refuses (page >= 1); reject
    # it here with a clear error instead of a ValidationError at extraction.
    with pytest.raises(ValueError, match="below 1"):
        chunk_document("doc", "--- Page 0 ---\nbody")


def test_duplicate_page_marker_rejected():
    # Two "Page 1" chunks would share the chunk id "doc:p1" and make
    # provenance ambiguous.
    with pytest.raises(ValueError, match="repeats a page"):
        chunk_document("doc", "--- Page 1 ---\none\n--- Page 1 ---\ntwo")


def test_load_documents_missing_dir_raises(tmp_path):
    # A typo'd data dir must fail loudly, not build an empty graph that
    # answers "I cannot answer that" to everything.
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "no-such-dir")
