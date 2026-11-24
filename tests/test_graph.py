"""Graph tests: build rules, schema enforcement, traversal."""

from pathlib import Path

import pytest

from agent_graph_rag.chunking import load_chunks
from agent_graph_rag.extractor import StubExtractor
from agent_graph_rag.graph import KnowledgeGraph
from agent_graph_rag.schema import (
    Entity,
    EntityType,
    Provenance,
    Relationship,
    RelationshipType,
)

DATA = Path(__file__).parent.parent / "src" / "agent_graph_rag" / "data"


def _prov():
    return Provenance(doc_id="d", chunk_id="d:p1", page=1)


def _entity(name, type_, **attrs):
    return Entity(name=name, type=type_, attributes=attrs, provenance=_prov())


def _graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    for chunk in load_chunks(DATA):
        g.add_extraction(StubExtractor().extract(chunk))
    return g


def test_full_graph_size():
    g = _graph()
    assert len(g) == 25
    assert g.num_relationships == 24


def test_entities_by_type():
    g = _graph()
    assert sorted(g.entities_by_type(EntityType.COMPOUND)) == [
        "AZT-101", "BRX-207", "CLT-558"]
    assert len(g.entities_by_type(EntityType.PROPERTY)) == 14


def test_repeated_entity_merges_attributes():
    g = KnowledgeGraph()
    g.add_entity(_entity("R-14", EntityType.REACTION, kind="Suzuki"))
    g.add_entity(_entity("R-14", EntityType.REACTION, yield_pct="78"))
    assert g.entity("R-14")["attributes"] == {"kind": "Suzuki", "yield_pct": "78"}
    assert len(g) == 1


def test_relationship_with_missing_endpoint_rejected():
    g = KnowledgeGraph()
    g.add_entity(_entity("AZT-101", EntityType.COMPOUND))
    with pytest.raises(ValueError):
        g.add_relationship(Relationship(
            source="AZT-101", target="NOPE",
            type=RelationshipType.HAS_PROPERTY, provenance=_prov()))


def test_schema_violation_rejected():
    g = KnowledgeGraph()
    g.add_entity(_entity("AZT-101", EntityType.COMPOUND))
    g.add_entity(_entity("BRX-207", EntityType.COMPOUND))
    # Compound -PRODUCES-> Compound is not an allowed triple.
    with pytest.raises(ValueError):
        g.add_relationship(Relationship(
            source="AZT-101", target="BRX-207",
            type=RelationshipType.PRODUCES, provenance=_prov()))


def test_duplicate_relationship_not_double_counted():
    g = _graph()
    before = g.num_relationships
    g.add_relationship(Relationship(
        source="R-14", target="AZT-101", type=RelationshipType.PRODUCES,
        provenance=_prov()))
    assert g.num_relationships == before


def test_find_paths_two_hops():
    g = _graph()
    paths = g.find_paths("R-14", "solubility:AZT-101")
    assert len(paths) == 1
    assert [s.relationship for s in paths[0].steps] == ["PRODUCES", "HAS_PROPERTY"]


def test_find_paths_unknown_node_is_empty():
    assert _graph().find_paths("R-14", "NOPE") == []


def test_multi_hop_from_reaction():
    g = _graph()
    paths = g.multi_hop("R-14", max_hops=2)
    targets = {p.steps[-1].target for p in paths}
    assert "AZT-101" in targets and "solubility:AZT-101" in targets


def test_property_of():
    g = _graph()
    prop = g.property_of("AZT-101", "solubility")
    assert prop["value"] == "12.4" and prop["unit"] == "mg/mL"
    ic50 = g.property_of("AZT-101", "IC50")
    assert ic50["value"] == "310" and ic50["target"] == "Kinase-7"
    assert g.property_of("NOPE", "solubility") is None


def test_compounds_tested_against_ranks():
    g = _graph()
    tested = g.compounds_tested_against("Kinase-7")
    by_compound = {t["compound"]: t["ic50_nM"] for t in tested}
    assert by_compound == {"BRX-207": "42", "AZT-101": "310", "CLT-558": "18"}
    assert g.compounds_tested_against("Kinase-7") and \
        g.compounds_tested_against("NoSuchEnzyme") == []


def test_path_provenance_points_at_real_chunks():
    g = _graph()
    paths = g.find_paths("R-31", "melting_point:CLT-558")
    prov = paths[0].provenance
    assert {p.doc_id for p in prov} == {
        "lab-note-04-reaction-conditions", "lab-note-05-stability"}
