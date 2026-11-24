"""Export the knowledge graph to CSV and Cypher.

CSV: ``nodes.csv`` (name, type, attributes as JSON, provenance) and
``edges.csv`` (source, relationship, target, provenance). Cypher: one
``CREATE`` per node and one ``CREATE ... -[:TYPE]-> ...`` per relationship,
using ``MERGE``-safe labels, so the graph can be loaded into Neo4j.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from .graph import KnowledgeGraph


def _prov_str(prov: dict) -> str:
    return f"{prov.get('doc_id')}:{prov.get('chunk_id')}:p{prov.get('page')}"


def export_csv(graph: KnowledgeGraph, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = out_dir / "nodes.csv"
    edges_path = out_dir / "edges.csv"
    with nodes_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "type", "attributes_json", "provenance"])
        for name, data in sorted(graph.graph.nodes(data=True)):
            writer.writerow([
                name, data.get("type", ""),
                json.dumps(data.get("attributes", {}), sort_keys=True),
                _prov_str(data.get("provenance", {})),
            ])
    with edges_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "relationship", "target", "provenance"])
        for src, tgt, data in sorted(graph.graph.edges(data=True)):
            for rel_type in data.get("types", []):
                writer.writerow([
                    src, rel_type, tgt,
                    _prov_str(data.get("provenance", {}).get(rel_type, {})),
                ])
    return {"nodes": nodes_path, "edges": edges_path}


def _cypher_label(type_name: str) -> str:
    return "".join(ch for ch in type_name if ch.isalnum())


def _cypher_escape(value: str) -> str:
    return (value.replace("\\", "\\\\").replace("'", "\\'")
                 .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"))


_CYPHER_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _cypher_key(key: str) -> str:
    """A property key, backtick-quoted unless it is a plain identifier.

    LLM-extracted attribute keys can contain spaces ("melting point");
    unquoted they produce invalid Cypher.
    """
    if _CYPHER_KEY_RE.fullmatch(key):
        return key
    return "`" + key.replace("`", "``") + "`"


def export_cypher(graph: KnowledgeGraph) -> str:
    lines = ["// Knowledge graph exported by agent-graph-rag.",
             "// Load with: cypher-shell -f graph.cypher (Neo4j 5+)"]
    for name, data in sorted(graph.graph.nodes(data=True)):
        label = _cypher_label(data.get("type", "Entity"))
        attrs = dict(data.get("attributes", {}))
        # The node's name wins: an attribute literally called "name" must
        # not overwrite the node's identity in the exported statement.
        props = {"name": name,
                 **{k: v for k, v in attrs.items() if k != "name"}}
        props_str = ", ".join(
            f"{_cypher_key(k)}: '{_cypher_escape(str(v))}'" for k, v in props.items())
        lines.append(f"CREATE (:{label} {{{props_str}}});")
    for src, tgt, data in sorted(graph.graph.edges(data=True)):
        for rel_type in data.get("types", []):
            lines.append(
                f"MATCH (a {{name: '{_cypher_escape(src)}'}}), "
                f"(b {{name: '{_cypher_escape(tgt)}'}}) "
                f"CREATE (a)-[:{rel_type}]->(b);")
    return "\n".join(lines) + "\n"


def write_cypher(graph: KnowledgeGraph, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(export_cypher(graph))
    return path
