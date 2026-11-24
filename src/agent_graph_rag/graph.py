"""The directed knowledge graph (networkx) and multi-hop traversal.

Nodes are entities, edges are typed relationships, and every node and edge
carries provenance. The graph refuses schema violations: a relationship
whose (source type, type, target type) is not in ALLOWED_TRIPLES is
rejected instead of silently stored.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from .schema import (
    ALLOWED_TRIPLES,
    Entity,
    EntityType,
    GraphExtraction,
    Provenance,
    Relationship,
    RelationshipType,
)


@dataclass
class PathStep:
    source: str
    relationship: str
    target: str
    provenance: Provenance

    def render(self) -> str:
        p = self.provenance
        return (f"{self.source} -[{self.relationship}]-> {self.target} "
                f"({p.doc_id}, {p.chunk_id}, p.{p.page})")


@dataclass
class GraphPath:
    steps: list[PathStep]

    def render(self) -> str:
        return " | ".join(s.render() for s in self.steps)

    @property
    def provenance(self) -> list[Provenance]:
        return [s.provenance for s in self.steps]


class KnowledgeGraph:
    """Directed graph of validated entities with provenance everywhere."""

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()

    # -- build -----------------------------------------------------------
    def add_entity(self, entity: Entity) -> None:
        if self.graph.has_node(entity.name):
            existing = self.graph.nodes[entity.name]
            merged = dict(existing.get("attributes", {}))
            merged.update(entity.attributes)
            self.graph.nodes[entity.name]["attributes"] = merged
            return
        self.graph.add_node(
            entity.name,
            type=entity.type.value,
            attributes=dict(entity.attributes),
            provenance=entity.provenance.model_dump(),
        )

    def add_relationship(self, rel: Relationship) -> None:
        if not self.graph.has_node(rel.source) or not self.graph.has_node(rel.target):
            raise ValueError(
                f"Cannot add {rel.source} -{rel.type}-> {rel.target}: "
                "endpoint missing from graph")
        src_type = EntityType(self.graph.nodes[rel.source]["type"])
        tgt_type = EntityType(self.graph.nodes[rel.target]["type"])
        if (src_type, rel.type, tgt_type) not in ALLOWED_TRIPLES:
            raise ValueError(
                f"Schema violation: {src_type.value} -{rel.type.value}-> "
                f"{tgt_type.value} is not an allowed triple")
        if self.graph.has_edge(rel.source, rel.target):
            existing = self.graph[rel.source][rel.target]
            if rel.type.value not in existing.get("types", []):
                existing["types"].append(rel.type.value)
                existing["provenance"][rel.type.value] = rel.provenance.model_dump()
            return
        self.graph.add_edge(
            rel.source, rel.target,
            types=[rel.type.value],
            provenance={rel.type.value: rel.provenance.model_dump()},
        )

    def add_extraction(self, extraction: GraphExtraction) -> None:
        for entity in extraction.entities:
            self.add_entity(entity)
        for rel in extraction.relationships:
            self.add_relationship(rel)

    # -- reads -----------------------------------------------------------
    def __len__(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    @property
    def num_relationships(self) -> int:
        return sum(len(d.get("types", [])) for _, _, d in self.graph.edges(data=True))

    def entity(self, name: str) -> dict | None:
        return self.graph.nodes[name] if self.graph.has_node(name) else None

    def has_node(self, name: str) -> bool:
        return self.graph.has_node(name)

    def entities_by_type(self, type_: EntityType) -> list[str]:
        return [n for n, d in self.graph.nodes(data=True)
                if d.get("type") == type_.value]

    def neighbors(self, name: str, rel_type: RelationshipType | None = None,
                  direction: str = "out") -> list[tuple[str, str]]:
        """(neighbor, relationship type) pairs. direction: out | in | both."""
        out: list[tuple[str, str]] = []
        if direction in ("out", "both"):
            for tgt, d in self.graph[name].items():
                for t in d.get("types", []):
                    if rel_type is None or t == rel_type.value:
                        out.append((tgt, t))
        if direction in ("in", "both"):
            for src, d in self.graph.pred[name].items():
                for t in d.get("types", []):
                    if rel_type is None or t == rel_type.value:
                        out.append((src, t))
        return out

    def find_paths(self, source: str, target: str, max_hops: int = 3) -> list[GraphPath]:
        """All simple directed paths source -> target up to max_hops."""
        paths: list[GraphPath] = []
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            return paths
        try:
            raw_paths = list(nx.all_simple_paths(
                self.graph, source, target, cutoff=max_hops))
        except nx.NetworkXNoPath:
            return paths
        for nodes in raw_paths:
            steps = []
            for a, b in zip(nodes, nodes[1:]):
                edge = self.graph[a][b]
                rel_type = edge["types"][0]
                prov = Provenance(**edge["provenance"][rel_type])
                steps.append(PathStep(a, rel_type, b, prov))
            paths.append(GraphPath(steps))
        return paths

    def multi_hop(self, start: str, max_hops: int = 2) -> list[GraphPath]:
        """Every path of length 1..max_hops starting at ``start`` (BFS)."""
        paths: list[GraphPath] = []
        if not self.graph.has_node(start):
            return paths
        frontier: list[tuple[str, list[PathStep]]] = [(start, [])]
        for _hop in range(max_hops):
            nxt: list[tuple[str, list[PathStep]]] = []
            for node, steps in frontier:
                for tgt, rel_type in self.neighbors(node, direction="out"):
                    edge = self.graph[node][tgt]
                    prov = Provenance(**edge["provenance"][rel_type])
                    new_steps = steps + [PathStep(node, rel_type, tgt, prov)]
                    paths.append(GraphPath(new_steps))
                    nxt.append((tgt, new_steps))
            frontier = nxt
        return paths

    def property_of(self, compound: str, kind: str,
                    target: str | None = None) -> dict | None:
        """The Property node of ``kind`` linked from ``compound``, if any.

        ``target`` narrows IC50-style properties to one enzyme: a compound
        tested against several enzymes has one IC50 node per enzyme, and
        returning the first one regardless of target answers with the
        wrong enzyme's value.
        """
        if not self.graph.has_node(compound):
            return None
        for target_node, rel in self.neighbors(compound, RelationshipType.HAS_PROPERTY):
            attrs = self.graph.nodes[target_node].get("attributes", {})
            if attrs.get("kind") != kind:
                continue
            if target is not None and attrs.get("target") != target:
                continue
            return {"name": target_node, **attrs,
                    "provenance": self.graph.nodes[target_node]["provenance"]}
        return None

    def compounds_tested_against(self, enzyme: str) -> list[dict]:
        out = []
        for compound in self.entities_by_type(EntityType.COMPOUND):
            for target, rel in self.neighbors(compound, RelationshipType.TESTED_AGAINST):
                if target == enzyme:
                    ic50 = self.property_of(compound, "IC50", target=enzyme)
                    out.append({"compound": compound,
                                "ic50_nM": (ic50 or {}).get("value"),
                                "ic50_target": (ic50 or {}).get("target")})
        return out
