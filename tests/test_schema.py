"""Schema validation tests: nothing unvalidated enters the graph."""

import pytest

from agent_graph_rag.schema import (
    ALLOWED_TRIPLES,
    Entity,
    EntityType,
    GraphExtraction,
    Provenance,
    Relationship,
    RelationshipType,
)


def _prov() -> Provenance:
    return Provenance(doc_id="d1", chunk_id="d1:p1", page=1, text_span="x")


def _entity(name="AZT-101", type_=EntityType.COMPOUND, **attrs):
    return Entity(name=name, type=type_, attributes=attrs, provenance=_prov())


def test_entity_validates():
    e = _entity()
    assert e.name == "AZT-101" and e.type == EntityType.COMPOUND


def test_entity_empty_name_rejected():
    with pytest.raises(ValueError):
        Entity(name="", type=EntityType.COMPOUND, provenance=_prov())


def test_property_without_kind_rejected():
    with pytest.raises(ValueError):
        _entity(name="p1", type_=EntityType.PROPERTY, value="12.4")


def test_property_with_kind_ok():
    e = _entity(name="solubility:AZT-101", type_=EntityType.PROPERTY,
                kind="solubility", value="12.4", unit="mg/mL")
    assert e.attributes["kind"] == "solubility"


def test_provenance_page_must_be_positive():
    with pytest.raises(ValueError):
        Provenance(doc_id="d", chunk_id="d:p0", page=0)


def test_relationship_self_loop_rejected():
    with pytest.raises(ValueError):
        Relationship(source="AZT-101", target="AZT-101",
                     type=RelationshipType.HAS_PROPERTY, provenance=_prov())


def test_extraction_rejects_dangling_relationship():
    with pytest.raises(ValueError):
        GraphExtraction(
            entities=[_entity()],
            relationships=[Relationship(source="AZT-101", target="MISSING",
                                        type=RelationshipType.HAS_PROPERTY,
                                        provenance=_prov())])


def test_extraction_accepts_valid_pair():
    ext = GraphExtraction(
        entities=[_entity(),
                  _entity(name="solubility:AZT-101", type_=EntityType.PROPERTY,
                          kind="solubility", value="12.4", unit="mg/mL")],
        relationships=[Relationship(source="AZT-101",
                                    target="solubility:AZT-101",
                                    type=RelationshipType.HAS_PROPERTY,
                                    provenance=_prov())])
    assert len(ext.entities) == 2 and len(ext.relationships) == 1


def test_allowed_triples_cover_the_four_relationships():
    rels = {t[1] for t in ALLOWED_TRIPLES}
    assert rels == {RelationshipType.PRODUCES, RelationshipType.CATALYZED_BY,
                    RelationshipType.HAS_PROPERTY, RelationshipType.TESTED_AGAINST}
