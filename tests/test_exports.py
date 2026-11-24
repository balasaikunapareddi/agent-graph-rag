"""Export tests: CSV nodes/edges and Cypher."""

import csv

from agent_graph_rag.agent import GraphRAGAgent, default_data_dir
from agent_graph_rag.exporters import export_csv, export_cypher, write_cypher
from agent_graph_rag.extractor import StubExtractor
from agent_graph_rag.graph import KnowledgeGraph
from agent_graph_rag.schema import Entity, EntityType, Provenance


def _prov():
    return Provenance(doc_id="d", chunk_id="d:p1", page=1)


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


def test_cypher_attribute_named_name_does_not_override_node_name():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="AZT-101", type=EntityType.COMPOUND,
                        attributes={"name": "EVIL"}, provenance=_prov()))
    cypher = export_cypher(g)
    assert "name: 'AZT-101'" in cypher
    assert "EVIL" not in cypher


def test_cypher_attribute_key_with_space_is_backticked():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="C1", type=EntityType.COMPOUND,
                        attributes={"melting point": "167"}, provenance=_prov()))
    assert "`melting point`: '167'" in export_cypher(g)


def test_cypher_escapes_quotes_backslashes_and_newlines():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="C1", type=EntityType.COMPOUND,
                        attributes={"description": "it's \\ tricky\nline"},
                        provenance=_prov()))
    cypher = export_cypher(g)
    create_lines = [ln for ln in cypher.splitlines() if "C1" in ln]
    assert len(create_lines) == 1  # no literal newline inside the statement
    assert "it\\'s \\\\ tricky\\nline" in create_lines[0]
