"""Agent + CLI tests: end-to-end comparison and the one-command demo."""

from agent_graph_rag.agent import DEMO_QUESTIONS, GraphRAGAgent, default_data_dir
from agent_graph_rag.cli import main, run_demo
from agent_graph_rag.extractor import StubExtractor


def test_compare_graph_beats_baseline():
    agent = GraphRAGAgent(extractor=StubExtractor())
    agent.ingest(default_data_dir())
    comparisons = agent.run_demo_questions()
    assert len(comparisons) == 5
    assert sum(c.graph_correct for c in comparisons) == 5
    assert sum(c.baseline_correct for c in comparisons) == 0


def test_compare_marks_expected_substrings():
    agent = GraphRAGAgent(extractor=StubExtractor())
    agent.ingest(default_data_dir())
    comp = agent.compare(DEMO_QUESTIONS[1]["question"],
                         DEMO_QUESTIONS[1]["expected"])
    assert comp.graph_correct
    assert not comp.baseline_correct  # baseline finds catalyst, misses yield


def test_demo_returns_zero(capsys):
    assert run_demo() == 0
    out = capsys.readouterr().out
    assert "GraphRAG: 5/5" in out and "Baseline: 0/5" in out
    assert "AZT-101" in out and "12.4" in out


def test_demo_with_exports(tmp_path):
    assert run_demo(export_dir=tmp_path) == 0
    assert (tmp_path / "nodes.csv").exists()
    assert (tmp_path / "edges.csv").exists()
    assert (tmp_path / "graph.cypher").exists()


def test_cli_demo_command(capsys):
    assert main(["demo"]) == 0
    assert "Score" in capsys.readouterr().out
