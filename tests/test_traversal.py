"""Traversal answering tests: every demo question, answered by the graph."""

import sys
import types
from pathlib import Path

import pytest

from agent_graph_rag.agent import DEMO_QUESTIONS, GraphRAGAgent, default_data_dir
from agent_graph_rag.extractor import StubExtractor
from agent_graph_rag.schema import (
    Entity,
    EntityType,
    Provenance,
    Relationship,
    RelationshipType,
)

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


def test_ic50_question_uses_named_enzyme(agent):
    # BRX-207: 42 nM vs Kinase-7 but 1850 nM vs Phosphatase-2. The answer
    # must be the named enzyme's value, not whichever IC50 was stored first.
    answer = agent.answer("What was the IC50 of BRX-207 against Phosphatase-2?")
    assert "1850" in answer.text
    assert "42" not in answer.text


def test_ic50_ranking_against_phosphatase(agent):
    answer = agent.answer("Which compound had the lowest IC50 against "
                          "Phosphatase-2, and which reaction produced it?")
    assert "BRX-207" in answer.text and "1850" in answer.text
    assert "R-22" in answer.text


def test_logp_answer_has_no_double_period(agent):
    assert agent.answer("What was the logP of AZT-101?").text == \
        "AZT-101 has logP 2.1."


def _add_tested_compound(agent, name, ic50_value):
    prov = Provenance(doc_id="d", chunk_id="d:p1", page=1)
    agent.graph.add_entity(Entity(name=name, type=EntityType.COMPOUND,
                                  provenance=prov))
    agent.graph.add_entity(Entity(name="Kinase-7", type=EntityType.ENZYME,
                                  provenance=prov))
    agent.graph.add_entity(Entity(
        name=f"IC50:Kinase-7:{name}", type=EntityType.PROPERTY,
        attributes={"kind": "IC50", "value": ic50_value, "unit": "nM",
                    "target": "Kinase-7"}, provenance=prov))
    agent.graph.add_relationship(Relationship(
        source=name, target="Kinase-7", type=RelationshipType.TESTED_AGAINST,
        provenance=prov))
    agent.graph.add_relationship(Relationship(
        source=name, target=f"IC50:Kinase-7:{name}",
        type=RelationshipType.HAS_PROPERTY, provenance=prov))


def test_ic50_ranking_ignores_non_numeric_values():
    # LLM-extracted IC50s can be non-numeric (">1000"); ranking must skip
    # them instead of crashing the answer.
    agent = GraphRAGAgent(extractor=StubExtractor())
    agent.ingest(DATA)
    _add_tested_compound(agent, "ZZZ-999", ">1000")
    answer = agent.answer("Which compound had the lowest IC50 against Kinase-7, "
                          "and which reaction produced it?")
    assert "CLT-558" in answer.text and "18" in answer.text


def test_ic50_ranking_all_non_numeric_is_not_guessed():
    agent = GraphRAGAgent(extractor=StubExtractor())
    _add_tested_compound(agent, "ZZZ-999", ">1000")
    answer = agent.answer("Which compound had the lowest IC50 against Kinase-7?")
    assert not answer.complete
    assert "cannot answer" in answer.text.lower()


def test_llm_synthesize_falls_back_on_api_error(agent, monkeypatch):
    # A failed rephrase must not destroy the good deterministic answer.
    module = types.ModuleType("litellm")

    def _boom(*args, **kwargs):
        raise RuntimeError("503 Service Unavailable")

    module.completion = _boom
    monkeypatch.setitem(sys.modules, "litellm", module)
    answer = agent.answer(DEMO_QUESTIONS[0]["question"], use_llm=True)
    assert answer.complete
    assert "AZT-101" in answer.text and "12.4" in answer.text
