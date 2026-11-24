"""Traversal answering tests: every demo question, answered by the graph."""

from pathlib import Path

import pytest

from agent_graph_rag.agent import DEMO_QUESTIONS, GraphRAGAgent, default_data_dir
from agent_graph_rag.extractor import StubExtractor

DATA = default_data_dir()


@pytest.fixture(scope="module")
def agent():
    a = GraphRAGAgent(extractor=StubExtractor())
    a.ingest(DATA)
    return a


def test_ingest_stats(agent):
    assert len(agent.graph) == 25
    assert len(agent.chunks) == 10


@pytest.mark.parametrize("item", DEMO_QUESTIONS, ids=[f"q{i}" for i in range(1, 6)])
def test_demo_questions_answered_correctly(agent, item):
    answer = agent.answer(item["question"])
    assert answer.complete
    for expected in item["expected"]:
        assert expected in answer.text


def test_answer_carries_provenance(agent):
    answer = agent.answer(DEMO_QUESTIONS[0]["question"])
    lines = answer.provenance_lines()
    assert len(lines) >= 2
    assert any("lab-note-01-synthesis" in line for line in lines)
    assert any("lab-note-03-properties" in line for line in lines)


def test_unknown_question_is_not_guessed(agent):
    answer = agent.answer("What is the boiling point of compound ZZZ-999?")
    assert not answer.complete
    assert "cannot answer" in answer.text.lower()


def test_ic50_ranking_picks_lowest(agent):
    answer = agent.answer("Which compound had the lowest IC50 against Kinase-7, "
                          "and which reaction produced it?")
    assert "CLT-558" in answer.text and "18" in answer.text and "R-31" in answer.text


def test_catalyst_and_yield_together(agent):
    answer = agent.answer("What catalyst was used in the reaction that produced "
                          "BRX-207, and what was the yield?")
    assert "HATU" in answer.text and "64%" in answer.text


def test_paths_render_with_provenance(agent):
    answer = agent.answer(DEMO_QUESTIONS[3]["question"])
    rendered = " ".join(p.render() for p in answer.paths)
    assert "PRODUCES" in rendered and "HAS_PROPERTY" in rendered
    assert "lab-note-04-reaction-conditions" in rendered
