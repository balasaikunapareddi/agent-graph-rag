"""Entity/relationship extraction: LLM (via LiteLLM) and deterministic stub.

Two backends share one interface so offline tests and the one-command demo
run with no key, while the live test uses real Gemini:

- :class:`StubExtractor` — deterministic regex extraction tuned to the
  shipped lab-note phrasing. No network, no key, fully reproducible.
- :class:`LLMExtractor` — calls the model named by ``LLM_MODEL``
  (default ``gemini/gemini-3.5-flash-lite``) with the extraction system
  prompt, parses the JSON reply, and validates it through the Pydantic
  schema. One retry on a malformed reply; then it raises.

Both return :class:`~agent_graph_rag.schema.GraphExtraction` with provenance
on every entity and relationship, stamped from the chunk being read.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod

from .chunking import Chunk
from .prompts import EXTRACTION_SYSTEM_PROMPT
from .schema import (
    Entity,
    EntityType,
    GraphExtraction,
    Provenance,
    Relationship,
    RelationshipType,
)

DEFAULT_MODEL = "gemini/gemini-3.5-flash-lite"

# --- regexes for the stub extractor (match the shipped lab-note phrasing) ---
REACTION_RE = re.compile(r"\b(R-\d+)\b")
COMPOUND_RE = re.compile(r"\b([A-Z]{2,4}-\d+)\b")
PRODUCED_RE = re.compile(r"Reaction (R-\d+) produced compound ([A-Z]{2,4}-\d+)")
CATALYST_RE = re.compile(r"[Tt]he catalyst was ([A-Za-z][A-Za-z0-9()]+)")
KIND_RE = re.compile(r"Reaction (R-\d+) was run as (?:a|an) ([^.,]+)")
YIELD_RE = re.compile(r"with a yield of (\d+)%")
SOLUBILITY_RE = re.compile(
    r"[Tt]he solubility of ([A-Z]{2,4}-\d+) was ([\d.]+) mg/mL")
MELTING_RE = re.compile(
    r"[Tt]he melting point of ([A-Z]{2,4}-\d+) was (\d+) °C")
LOGP_RE = re.compile(r"[Tt]he logP of ([A-Z]{2,4}-\d+) was ([\d.]+)")
ASSAY_RE = re.compile(
    r"([A-Z]{2,4}-\d+) was tested against enzyme ([A-Za-z]+-\d+) "
    r"with an IC50 of ([\d,]+) nM")
STABLE_RE = re.compile(
    r"([A-Z]{2,4}-\d+) remained stable for ([^.]+?)(?: with no detectable degradation)?\.")
DESCRIPTION_RE = re.compile(
    r"Compound ([A-Z]{2,4}-\d+) is (?:a|an) ([^.]+)\.")


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, chunk: Chunk) -> GraphExtraction:
        raise NotImplementedError


class StubExtractor(BaseExtractor):
    """Deterministic regex extractor. No LLM, no key, no network."""

    def extract(self, chunk: Chunk) -> GraphExtraction:
        prov = Provenance(text_span=_first_sentence(chunk.text), **chunk.provenance_kwargs)
        entities: dict[str, Entity] = {}
        relationships: list[Relationship] = []

        def add_entity(name: str, type_: EntityType, **attrs: str) -> None:
            if name in entities:
                entities[name].attributes.update(
                    {k: v for k, v in attrs.items() if v})
                return
            entities[name] = Entity(
                name=name, type=type_, attributes=dict(attrs), provenance=prov)

        def add_rel(source: str, target: str, type_: RelationshipType) -> None:
            if not any(r.source == source and r.target == target
                       and r.type == type_ for r in relationships):
                relationships.append(Relationship(
                    source=source, target=target, type=type_, provenance=prov))

        text = chunk.text
        for m in PRODUCED_RE.finditer(text):
            reaction, compound = m.group(1), m.group(2)
            add_entity(reaction, EntityType.REACTION)
            add_entity(compound, EntityType.COMPOUND)
            add_rel(reaction, compound, RelationshipType.PRODUCES)
        for m in KIND_RE.finditer(text):
            add_entity(m.group(1), EntityType.REACTION,
                       kind=m.group(2).strip())
        for m in CATALYST_RE.finditer(text):
            reaction = REACTION_RE.search(text)
            if reaction:
                add_entity(reaction.group(1), EntityType.REACTION)
                add_entity(m.group(1), EntityType.CATALYST)
                add_rel(reaction.group(1), m.group(1), RelationshipType.CATALYZED_BY)
        for m in YIELD_RE.finditer(text):
            reaction = REACTION_RE.search(text)
            if reaction:
                add_entity(reaction.group(1), EntityType.REACTION,
                           yield_pct=m.group(1))
        for m in DESCRIPTION_RE.finditer(text):
            add_entity(m.group(1), EntityType.COMPOUND,
                       description=m.group(2).strip())
        for m in SOLUBILITY_RE.finditer(text):
            compound, value = m.group(1), m.group(2)
            prop = f"solubility:{compound}"
            add_entity(compound, EntityType.COMPOUND)
            add_entity(prop, EntityType.PROPERTY, kind="solubility",
                       value=value, unit="mg/mL")
            add_rel(compound, prop, RelationshipType.HAS_PROPERTY)
        for m in MELTING_RE.finditer(text):
            compound, value = m.group(1), m.group(2)
            prop = f"melting_point:{compound}"
            add_entity(compound, EntityType.COMPOUND)
            add_entity(prop, EntityType.PROPERTY, kind="melting point",
                       value=value, unit="°C")
            add_rel(compound, prop, RelationshipType.HAS_PROPERTY)
        for m in LOGP_RE.finditer(text):
            compound, value = m.group(1), m.group(2)
            prop = f"logP:{compound}"
            add_entity(compound, EntityType.COMPOUND)
            add_entity(prop, EntityType.PROPERTY, kind="logP",
                       value=value, unit="")
            add_rel(compound, prop, RelationshipType.HAS_PROPERTY)
        for m in ASSAY_RE.finditer(text):
            compound, enzyme, ic50 = m.group(1), m.group(2), m.group(3)
            prop = f"IC50:{enzyme}:{compound}"
            add_entity(compound, EntityType.COMPOUND)
            add_entity(enzyme, EntityType.ENZYME)
            add_entity(prop, EntityType.PROPERTY, kind="IC50",
                       value=ic50.replace(",", ""), unit="nM", target=enzyme)
            add_rel(compound, enzyme, RelationshipType.TESTED_AGAINST)
            add_rel(compound, prop, RelationshipType.HAS_PROPERTY)
        for m in STABLE_RE.finditer(text):
            compound, span = m.group(1), m.group(2).strip()
            prop = f"stability:{compound}"
            add_entity(compound, EntityType.COMPOUND)
            add_entity(prop, EntityType.PROPERTY, kind="stability",
                       value=span, unit="")
            add_rel(compound, prop, RelationshipType.HAS_PROPERTY)

        return GraphExtraction(
            entities=list(entities.values()), relationships=relationships)


def _first_sentence(text: str) -> str:
    skip_prefixes = ("#", "Date:", "Researcher:", "Method:", "Assay:", "Project:")
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith(skip_prefixes):
            return line
    return text[:120]


class LLMExtractor(BaseExtractor):
    """LLM extractor via LiteLLM. Model from ``LLM_MODEL`` env var."""

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)

    def extract(self, chunk: Chunk) -> GraphExtraction:
        import litellm

        user = (
            f"Document: {chunk.doc_id}\nChunk: {chunk.chunk_id}\n"
            f"Page: {chunk.page}\n\n{chunk.text}"
        )
        last_error: Exception | None = None
        for _attempt in range(2):
            try:
                resp = litellm.completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                )
            except Exception as exc:  # noqa: BLE001 - transient API error: retry once
                last_error = exc
                if _attempt == 0:
                    import time
                    time.sleep(20)
                    continue
                raise ValueError(f"LLM extraction API call failed: {exc}") from exc
            raw = resp.choices[0].message.content or ""
            try:
                data = json.loads(_strip_fences(raw))
                return _validate_llm_payload(data, chunk)
            except Exception as exc:  # noqa: BLE001 - retry once, then raise
                last_error = exc
        raise ValueError(f"LLM extraction failed to validate: {last_error}")


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return raw


def _validate_llm_payload(data: dict, chunk: Chunk) -> GraphExtraction:
    """Attach chunk provenance to every LLM-returned entity/relationship."""
    prov = Provenance(text_span=_first_sentence(chunk.text),
                      **chunk.provenance_kwargs)
    entities = [
        Entity(name=e["name"], type=e["type"],
               attributes={str(k): str(v) for k, v in e.get("attributes", {}).items()},
               provenance=prov)
        for e in data.get("entities", [])
    ]
    relationships = [
        Relationship(source=r["source"], target=r["target"], type=r["type"],
                     provenance=prov)
        for r in data.get("relationships", [])
    ]
    return GraphExtraction(entities=entities, relationships=relationships)
