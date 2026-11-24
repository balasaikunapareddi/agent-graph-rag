"""Extraction tests: stub extractor + LLM payload validation (no network)."""

from pathlib import Path

from agent_graph_rag.chunking import Chunk, load_chunks
from agent_graph_rag.extractor import (
    StubExtractor,
    _strip_fences,
    _validate_llm_payload,
)
from agent_graph_rag.schema import EntityType, RelationshipType

DATA = Path(__file__).parent.parent / "src" / "agent_graph_rag" / "data"

EXT = StubExtractor()


def _chunk(text, doc_id="d", page=1):
    return Chunk(doc_id=doc_id, chunk_id=f"{doc_id}:p{page}", page=page, text=text)


def test_stub_extracts_produces():
    ext = EXT.extract(_chunk("Reaction R-14 produced compound AZT-101 with a yield of 78%."))
    assert ("R-14", "AZT-101", RelationshipType.PRODUCES) in [
        (r.source, r.target, r.type) for r in ext.relationships]


def test_stub_extracts_catalyst_and_yield():
    ext = EXT.extract(_chunk("Reaction R-22 was run as an amide coupling. "
                             "The catalyst was HATU. Reaction R-22 produced "
                             "compound BRX-207 with a yield of 64%."))
    names = {e.name: e for e in ext.entities}
    assert names["HATU"].type == EntityType.CATALYST
    assert names["R-22"].attributes["yield_pct"] == "64"


def test_stub_extracts_property_with_value_and_unit():
    ext = EXT.extract(_chunk("The solubility of AZT-101 was 12.4 mg/mL. "
                             "The melting point of AZT-101 was 184 °C."))
    props = {e.name: e for e in ext.entities if e.type == EntityType.PROPERTY}
    assert props["solubility:AZT-101"].attributes["value"] == "12.4"
    assert props["melting_point:AZT-101"].attributes["unit"] == "°C"


def test_stub_extracts_assay_and_ic50_property():
    ext = EXT.extract(_chunk("BRX-207 was tested against enzyme Kinase-7 "
                             "with an IC50 of 42 nM."))
    rels = {(r.source, r.target, r.type) for r in ext.relationships}
    assert ("BRX-207", "Kinase-7", RelationshipType.TESTED_AGAINST) in rels
    assert ("BRX-207", "IC50:Kinase-7:BRX-207", RelationshipType.HAS_PROPERTY) in rels


def test_stub_empty_chunk_extracts_nothing():
    ext = EXT.extract(_chunk("Both compounds were screened in duplicate."))
    assert ext.entities == [] and ext.relationships == []


def test_stub_provenance_on_everything():
    chunks = load_chunks(DATA)
    for chunk in chunks:
        ext = EXT.extract(chunk)
        for entity in ext.entities:
            assert entity.provenance.chunk_id == chunk.chunk_id
            assert entity.provenance.page == chunk.page
        for rel in ext.relationships:
            assert rel.provenance.doc_id == chunk.doc_id


def test_shipped_docs_yield_expected_entities():
    chunks = load_chunks(DATA)
    names = set()
    for chunk in chunks:
        names |= {e.name for e in EXT.extract(chunk).entities}
    for expected in ["AZT-101", "BRX-207", "CLT-558", "R-14", "R-22", "R-31",
                     "Kinase-7", "Phosphatase-2", "HATU", "Pd(PPh3)4", "NaBH3CN"]:
        assert expected in names


def test_strip_fences_removes_markdown():
    assert _strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_validate_llm_payload_stamps_provenance():
    chunk = _chunk("Reaction R-14 produced compound AZT-101.")
    data = {"entities": [
        {"name": "R-14", "type": "Reaction", "attributes": {}},
        {"name": "AZT-101", "type": "Compound", "attributes": {}}],
        "relationships": [
            {"source": "R-14", "target": "AZT-101", "type": "PRODUCES"}]}
    ext = _validate_llm_payload(data, chunk)
    assert len(ext.entities) == 2
    assert all(e.provenance.chunk_id == "d:p1" for e in ext.entities)


def test_validate_llm_payload_rejects_dangling_relationship():
    import pytest
    chunk = _chunk("x")
    data = {"entities": [{"name": "AZT-101", "type": "Compound"}],
            "relationships": [{"source": "AZT-101", "target": "NOPE",
                               "type": "HAS_PROPERTY"}]}
    with pytest.raises(ValueError):
        _validate_llm_payload(data, chunk)
