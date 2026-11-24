"""Baseline tests: plain chunk RAG on the same docs and questions."""

from agent_graph_rag.agent import DEMO_QUESTIONS
from agent_graph_rag.baseline import PlainChunkRAG
from agent_graph_rag.chunking import load_chunks
from agent_graph_rag.agent import default_data_dir

CHUNKS = load_chunks(default_data_dir())


def test_retrieve_finds_reaction_chunk():
    rag = PlainChunkRAG(CHUNKS)
    top, score = rag.retrieve("Which compound was produced by reaction R-14?")[0]
    assert top.chunk_id == "lab-note-01-synthesis:p1"
    assert score > 0


def test_retrieve_empty_for_unrelated_question():
    rag = PlainChunkRAG(CHUNKS)
    assert rag.retrieve("quantum entanglement in superconductors") == []


def test_baseline_partial_on_multi_hop():
    rag = PlainChunkRAG(CHUNKS)
    answer = rag.answer(DEMO_QUESTIONS[0]["question"])
    assert not answer.complete
    assert "different chunk" in answer.text
    assert "lab-note-01-synthesis:p1" in answer.retrieved_chunk_ids


def test_baseline_partial_on_melting_point_via_reaction():
    rag = PlainChunkRAG(CHUNKS)
    for q in (DEMO_QUESTIONS[3]["question"], DEMO_QUESTIONS[4]["question"]):
        assert not rag.answer(q).complete


def test_baseline_can_answer_single_chunk_fact():
    rag = PlainChunkRAG(CHUNKS)
    answer = rag.answer("What is the solubility of AZT-101?")
    assert answer.complete
    assert "12.4" in answer.text


def test_baseline_reports_retrieved_chunks():
    rag = PlainChunkRAG(CHUNKS)
    answer = rag.answer("What catalyst was used in reaction R-22?")
    assert answer.retrieved_chunk_ids
    assert "lab-note-01-synthesis:p2" in answer.retrieved_chunk_ids
