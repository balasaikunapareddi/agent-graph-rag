"""Export tests: CSV nodes/edges and Cypher."""

import csv

from agent_graph_rag.agent import GraphRAGAgent, default_data_dir
from agent_graph_rag.exporters import export_csv, export_cypher, write_cypher
from agent_graph_rag.extractor import StubExtractor


def _agent():
    agent = GraphRAGAgent(extractor=StubExtractor())
    agent.ingest(default_data_dir())
    return agent


def test_csv_nodes_cover_all_entities(tmp_path):
    paths = export_csv(_agent().graph, tmp_path)
    with paths["nodes"].open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 25
    assert {r["name"] for r in rows} >= {"AZT-101", "R-14", "Kinase-7"}
    assert all(r["provenance"] for r in rows)  # provenance on every node


def test_csv_edges_cover_all_relationships(tmp_path):
    paths = export_csv(_agent().graph, tmp_path)
    with paths["edges"].open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 24
    assert ("R-14", "PRODUCES", "AZT-101") in [
        (r["source"], r["relationship"], r["target"]) for r in rows]
    assert all(r["provenance"] for r in rows)


def test_cypher_has_nodes_and_relationships():
    cypher = export_cypher(_agent().graph)
    assert "CREATE (:Compound {name: 'AZT-101'" in cypher
    assert "CREATE (:Reaction {name: 'R-14'" in cypher
    assert "-[:PRODUCES]->" in cypher
    assert "-[:HAS_PROPERTY]->" in cypher
    assert "-[:TESTED_AGAINST]->" in cypher


def test_cypher_line_count_matches_graph():
    agent = _agent()
    cypher = export_cypher(agent.graph)
    creates = [ln for ln in cypher.splitlines() if ln.startswith("CREATE (")]
    matches = [ln for ln in cypher.splitlines() if ln.startswith("MATCH")]
    assert len(creates) == 25
    assert len(matches) == 24


def test_write_cypher_creates_file(tmp_path):
    path = write_cypher(_agent().graph, tmp_path / "sub" / "graph.cypher")
    assert path.exists() and "PRODUCES" in path.read_text()
