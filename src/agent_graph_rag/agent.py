"""GraphRAGAgent: ingest documents, build the graph, answer multi-hop questions.

The agent is deliberately small and explicit: extraction (LLM or stub) ->
validated Pydantic objects -> directed graph with provenance -> traversal
answering. No agent framework is used — for this build the core *is* the
graph and the extraction, and networkx + LiteLLM + Pydantic are the natural
fit; a framework would add indirection without adding capability. The
answering step is deterministic traversal over the graph, with an optional
LLM synthesis pass that may only rephrase the traversed paths, never add
facts. That split is what makes the answers checkable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .baseline import BaselineAnswer, PlainChunkRAG
from .chunking import Chunk, load_chunks
from .extractor import DEFAULT_MODEL, BaseExtractor, LLMExtractor, StubExtractor
from .graph import GraphPath, KnowledgeGraph
from .prompts import ANSWER_SYSTEM_PROMPT
from .schema import EntityType

REACTION_RE = re.compile(r"\b(R-\d+)\b")
COMPOUND_RE = re.compile(r"\b([A-Z]{2,4}-\d+)\b")
ENZYME_RE = re.compile(r"\b([A-Z][a-z]+-\d+)\b")

def _parse_number(value: object) -> float | None:
    """Float value of a stored IC50, or None if it is not numeric.

    LLM-extracted values are strings and may be non-numeric (">1000",
    "n/a"); ranking must skip those instead of crashing the answer.
    """
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


DEMO_QUESTIONS: list[dict] = [
    {"question": "Which compound was produced by reaction R-14, and what "
                 "solubility was measured for it?",
     "expected": ["AZT-101", "12.4"]},
    {"question": "What catalyst was used in the reaction that produced "
                 "BRX-207, and what was the yield?",
     "expected": ["HATU", "64"]},
    {"question": "Which compound had the lowest IC50 against Kinase-7, "
                 "and which reaction produced it?",
     "expected": ["CLT-558", "18", "R-31"]},
    {"question": "What was the melting point of the compound produced "
                 "by reaction R-31?",
     "expected": ["CLT-558", "167"]},
    {"question": "What was the melting point of the compound produced "
                 "by reaction R-14?",
     "expected": ["AZT-101", "184"]},
]


@dataclass
class GraphAnswer:
    text: str
    paths: list[GraphPath] = field(default_factory=list)
    complete: bool = True

    def provenance_lines(self) -> list[str]:
        seen: list[str] = []
        for path in self.paths:
            for step in path.steps:
                line = (f"{step.provenance.doc_id}, "
                        f"{step.provenance.chunk_id}, p.{step.provenance.page}")
                if line not in seen:
                    seen.append(line)
        return seen


@dataclass
class Comparison:
    question: str
    graph_answer: GraphAnswer
    baseline_answer: BaselineAnswer
    expected: list[str]

    @property
    def graph_correct(self) -> bool:
        return all(e in self.graph_answer.text for e in self.expected)

    @property
    def baseline_correct(self) -> bool:
        return self.baseline_answer.complete and all(
            e in self.baseline_answer.text for e in self.expected)


class GraphRAGAgent:
    """Build a knowledge graph from documents and answer by traversal."""

    def __init__(self, extractor: BaseExtractor | None = None,
                 model: str | None = None):
        self.extractor = extractor or StubExtractor()
        self.model = model
        self.graph = KnowledgeGraph()
        self.chunks: list[Chunk] = []
        self.baseline: PlainChunkRAG | None = None

    # -- ingest ----------------------------------------------------------
    def ingest(self, data_dir: str | Path) -> dict:
        self.chunks = load_chunks(data_dir)
        for chunk in self.chunks:
            self.graph.add_extraction(self.extractor.extract(chunk))
        self.baseline = PlainChunkRAG(self.chunks)
        return {"chunks": len(self.chunks), "nodes": len(self.graph),
                "edges": self.graph.num_edges,
                "relationships": self.graph.num_relationships}

    def ingest_chunks(self, chunks: list[Chunk]) -> dict:
        self.chunks = list(chunks)
        for chunk in self.chunks:
            self.graph.add_extraction(self.extractor.extract(chunk))
        self.baseline = PlainChunkRAG(self.chunks)
        return {"chunks": len(self.chunks), "nodes": len(self.graph),
                "edges": self.graph.num_edges,
                "relationships": self.graph.num_relationships}

    # -- answering -------------------------------------------------------
    def answer(self, question: str, use_llm: bool = False) -> GraphAnswer:
        result = self._traverse_answer(question)
        if use_llm and result.complete:
            result.text = self._llm_synthesize(question, result)
        return result

    def _traverse_answer(self, question: str) -> GraphAnswer:
        q = question.lower()
        reactions = REACTION_RE.findall(question)
        compounds = [c for c in COMPOUND_RE.findall(question)
                     if not c.startswith("R-")]
        enzymes = [e for e in ENZYME_RE.findall(question)
                   if e not in reactions and e not in compounds]

        # "lowest/highest IC50 against <enzyme>" — rank all tested compounds.
        if "ic50" in q and ("lowest" in q or "highest" in q):
            enzyme = enzymes[0] if enzymes else "Kinase-7"
            tested = [t for t in self.graph.compounds_tested_against(enzyme)
                      if t["ic50_nM"]]
            ranked = [(t, _parse_number(t["ic50_nM"])) for t in tested]
            ranked = [(t, v) for t, v in ranked if v is not None]
            if not ranked:
                return GraphAnswer("I cannot answer that from the documents provided.",
                                   complete=False)
            best, _value = (min(ranked, key=lambda tv: tv[1])
                            if "lowest" in q else max(ranked, key=lambda tv: tv[1]))
            compound = best["compound"]
            paths = self.graph.find_paths(compound, enzyme)
            producing = self.graph.neighbors(compound, direction="in")
            reaction = next((s for s, r in producing if r == "PRODUCES"), None)
            if reaction:
                paths += self.graph.find_paths(reaction, compound)
            return GraphAnswer(
                f"{compound} had the {'lowest' if 'lowest' in q else 'highest'} "
                f"IC50 against {enzyme} at {best['ic50_nM']} nM"
                + (f", and it was produced by reaction {reaction}."
                   if reaction else "."),
                paths=paths)

        # Resolve the starting compound: named directly, or via a reaction.
        compound = compounds[0] if compounds else None
        paths: list[GraphPath] = []
        if compound is None and reactions:
            reaction = reactions[0]
            produced = [t for t, r in self.graph.neighbors(reaction)
                        if r == "PRODUCES"]
            if not produced:
                return GraphAnswer("I cannot answer that from the documents provided.",
                                   complete=False)
            compound = produced[0]
            paths += self.graph.find_paths(reaction, compound)
        if compound is None:
            return GraphAnswer("I cannot answer that from the documents provided.",
                               complete=False)
        if not self.graph.has_node(compound):
            return GraphAnswer("I cannot answer that from the documents provided.",
                               complete=False)

        # Property intents.
        if "solubility" in q:
            prop = self.graph.property_of(compound, "solubility")
            return self._property_answer(question, compound, prop,
                                         "solubility", "mg/mL", paths)
        if "melting point" in q:
            prop = self.graph.property_of(compound, "melting point")
            return self._property_answer(question, compound, prop,
                                         "melting point", "°C", paths)
        if "logp" in q:
            prop = self.graph.property_of(compound, "logP")
            return self._property_answer(question, compound, prop, "logP", "", paths)
        if "ic50" in q:
            enzyme = enzymes[0] if enzymes else None
            prop = self.graph.property_of(compound, "IC50", target=enzyme)
            return self._property_answer(question, compound, prop,
                                         "IC50", "nM", paths)
        if "stability" in q or "stable" in q:
            prop = self.graph.property_of(compound, "stability")
            return self._property_answer(question, compound, prop,
                                         "stability", "", paths)

        # Catalyst / yield intents: go back through the producing reaction.
        if "catalyst" in q or "yield" in q:
            producing = self.graph.neighbors(compound, direction="in")
            reaction = next((s for s, r in producing if r == "PRODUCES"), None)
            if reaction is None:
                return GraphAnswer("I cannot answer that from the documents provided.",
                                   complete=False)
            paths += self.graph.find_paths(reaction, compound)
            catalysts = [t for t, r in self.graph.neighbors(reaction)
                         if r == "CATALYZED_BY"]
            reaction_attrs = self.graph.entity(reaction) or {}
            yield_pct = (reaction_attrs.get("attributes") or {}).get("yield_pct")
            bits = []
            if "catalyst" in q and catalysts:
                bits.append(f"the catalyst was {catalysts[0]}")
                paths += self.graph.find_paths(reaction, catalysts[0])
            if "yield" in q and yield_pct:
                bits.append(f"the yield was {yield_pct}%")
            if not bits:
                return GraphAnswer("I cannot answer that from the documents provided.",
                                   complete=False)
            return GraphAnswer(
                f"Reaction {reaction} produced {compound}; "
                + "; ".join(bits) + ".", paths=paths)

        # "Which compound was produced by reaction X" (no property asked).
        if reactions:
            return GraphAnswer(f"Reaction {reactions[0]} produced compound {compound}.",
                               paths=paths)
        return GraphAnswer(f"{compound} is in the graph; no property was asked for.",
                           paths=paths)

    def _property_answer(self, question: str, compound: str, prop: dict | None,
                         kind: str, unit: str, paths: list[GraphPath]) -> GraphAnswer:
        if prop is None:
            return GraphAnswer("I cannot answer that from the documents provided.",
                               complete=False)
        paths = paths + self.graph.find_paths(compound, prop["name"])
        value = " ".join(p for p in (str(prop.get("value", "")), unit) if p)
        text = f"{compound} has {kind} {value}."
        if "produced by reaction" in question.lower() or "reaction" in question.lower():
            reaction = next((s for s, r in
                             self.graph.neighbors(compound, direction="in")
                             if r == "PRODUCES"), None)
            if reaction:
                text = (f"Reaction {reaction} produced compound {compound}, "
                        f"and {compound} has {kind} {value}.")
        return GraphAnswer(text, paths=paths)

    def _llm_synthesize(self, question: str, result: GraphAnswer) -> str:
        """LLM may only rephrase the traversed paths — never add facts.

        A failed rephrase must not destroy a good deterministic answer:
        any API error falls back to the traversed text.
        """
        import litellm

        model = self.model or DEFAULT_MODEL
        paths_text = "\n".join(p.render() for p in result.paths)
        try:
            resp = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question: {question}\n\n"
                     f"Graph paths:\n{paths_text}\n\n"
                     f"Deterministic answer: {result.text}"},
                ],
                timeout=120,
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception:  # noqa: BLE001 - synthesis is cosmetic; keep the real answer
            return result.text
        return text or result.text

    # -- comparison ------------------------------------------------------
    def compare(self, question: str, expected: list[str],
                use_llm: bool = False) -> Comparison:
        graph_answer = self.answer(question, use_llm=use_llm)
        baseline_answer = (self.baseline.answer(question)
                           if self.baseline else BaselineAnswer(
                               "Baseline not built.", [], complete=False))
        return Comparison(question, graph_answer, baseline_answer, expected)

    def run_demo_questions(self, use_llm: bool = False) -> list[Comparison]:
        return [self.compare(q["question"], q["expected"], use_llm=use_llm)
                for q in DEMO_QUESTIONS]


def default_data_dir() -> Path:
    return Path(__file__).parent / "data"
