"""Extraction tests: stub extractor + LLM payload validation (no network)."""

import sys
import types
from pathlib import Path

import pytest

from agent_graph_rag.chunking import Chunk, load_chunks
from agent_graph_rag.extractor import (
    LLMExtractor,
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


def test_stub_logp_value_has_no_trailing_period():
    # "The logP of AZT-101 was 2.1." — the number regex must not swallow
    # the sentence's full stop into the stored value ("2.1.").
    ext = EXT.extract(_chunk("The logP of AZT-101 was 2.1."))
    props = {e.name: e for e in ext.entities if e.type == EntityType.PROPERTY}
    assert props["logP:AZT-101"].attributes["value"] == "2.1"


def test_stub_provenance_span_is_the_matched_fact():
    # Every entity's text_span must be the sentence it came from, not the
    # chunk's first line (melting point / logP used to cite solubility).
    ext = EXT.extract(_chunk("The solubility of AZT-101 was 12.4 mg/mL. "
                             "The melting point of AZT-101 was 184 °C. "
                             "The logP of AZT-101 was 2.1."))
    spans = {e.name: e.provenance.text_span for e in ext.entities}
    assert "solubility" in spans["solubility:AZT-101"]
    assert "melting point" in spans["melting_point:AZT-101"]
    assert "logP" in spans["logP:AZT-101"]


def _fake_litellm(monkeypatch, behaviours):
    """Stand-in litellm module: each call raises or returns per behaviour."""
    calls = {"n": 0, "kwargs": []}

    class _Msg:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Msg(content)

    class _Resp:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    def completion(model, messages, timeout=None):
        calls["n"] += 1
        calls["kwargs"].append({"model": model, "timeout": timeout})
        behaviour = behaviours[min(calls["n"] - 1, len(behaviours) - 1)]
        if isinstance(behaviour, Exception):
            raise behaviour
        return _Resp(behaviour)

    module = types.ModuleType("litellm")
    module.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", module)
    return calls


def _api_error(message, status_code):
    err = RuntimeError(message)
    err.status_code = status_code
    return err


_VALID_JSON = ('{"entities": ['
               '{"name": "R-14", "type": "Reaction", "attributes": {}}, '
               '{"name": "AZT-101", "type": "Compound", "attributes": {}}], '
               '"relationships": [{"source": "R-14", "target": "AZT-101", '
               '"type": "PRODUCES"}]}')


def test_llm_extractor_retries_transient_503_once(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    calls = _fake_litellm(monkeypatch, [
        _api_error("503 Service Unavailable", 503), _VALID_JSON])
    ext = LLMExtractor(model="fake/model", retry_delay=20).extract(
        _chunk("Reaction R-14 produced compound AZT-101."))
    assert calls["n"] == 2
    assert sleeps == [20]
    assert calls["kwargs"][0]["timeout"] == 120
    assert {e.name for e in ext.entities} == {"R-14", "AZT-101"}


def test_llm_extractor_does_not_retry_auth_error(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    calls = _fake_litellm(monkeypatch, [_api_error("401 Unauthorized", 401)])
    with pytest.raises(ValueError, match="API call failed"):
        LLMExtractor(model="fake/model").extract(_chunk("x"))
    assert calls["n"] == 1
    assert sleeps == []


def test_llm_extractor_retries_malformed_json_then_succeeds(monkeypatch):
    calls = _fake_litellm(
        monkeypatch, ["not json at all",
                      '{"entities": [], "relationships": []}'])
    ext = LLMExtractor(model="fake/model").extract(_chunk("x"))
    assert calls["n"] == 2
    assert ext.entities == [] and ext.relationships == []


def test_llm_extractor_malformed_json_twice_raises(monkeypatch):
    calls = _fake_litellm(monkeypatch, ["garbage", "still garbage"])
    with pytest.raises(ValueError, match="failed to validate"):
        LLMExtractor(model="fake/model").extract(_chunk("x"))
    assert calls["n"] == 2


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
