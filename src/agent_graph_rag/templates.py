"""Auto-generated entity/relationship templates from example documents.

Docling Graph auto-generates Pydantic templates from example docs; this
module does the same at small scale: run extraction over example chunks,
collect the entity types, relationship types, and attribute keys that
actually occur, and emit (a) a JSON template describing the schema and
(b) a Pydantic source file that compiles and validates entities of that
shape. The generated template is data, not hand-written code — re-running
it on new example docs produces a new template.
"""

from __future__ import annotations

import keyword
import re

from .chunking import Chunk
from .extractor import BaseExtractor
from .schema import EntityType, RelationshipType


def generate_template(chunks: list[Chunk],
                      extractor: BaseExtractor) -> dict:
    """Infer the schema template from what extraction finds in ``chunks``."""
    entity_types: dict[str, dict] = {}
    relationship_types: dict[str, dict] = {}
    for chunk in chunks:
        extraction = extractor.extract(chunk)
        for entity in extraction.entities:
            slot = entity_types.setdefault(entity.type.value, {"attributes": []})
            for key in entity.attributes:
                if key not in slot["attributes"]:
                    slot["attributes"].append(key)
        for rel in extraction.relationships:
            slot = relationship_types.setdefault(rel.type.value, {"pairs": []})
            pair = f"{_type_of(extraction, rel.source)}->{_type_of(extraction, rel.target)}"
            if pair not in slot["pairs"]:
                slot["pairs"].append(pair)
    return {
        "entity_types": entity_types,
        "relationship_types": relationship_types,
        "generated_from_chunks": len(chunks),
    }


def _type_of(extraction, name: str) -> str:
    for entity in extraction.entities:
        if entity.name == name:
            return entity.type.value
    return "Unknown"


def _safe_ident(name: str, used: set[str] | None = None) -> str:
    """A valid, unique Python identifier for an LLM-derived attribute key.

    Attribute keys come from extraction ("melting point", "class",
    "yield-pct") and are not identifiers; emitting them verbatim makes
    the generated template uncompilable. Sanitise, and de-duplicate
    ("yield-pct" and "yield_pct" would otherwise collide).
    """
    ident = re.sub(r"\W", "_", str(name))
    if not ident or ident[0].isdigit():
        ident = "_" + ident
    if keyword.iskeyword(ident):
        ident += "_"
    if used is not None:
        base, i = ident, 2
        while ident in used:
            ident = f"{base}_{i}"
            i += 1
        used.add(ident)
    return ident


def template_to_pydantic_code(template: dict, class_prefix: str = "Auto") -> str:
    """Render the template as importable Pydantic source code."""
    lines = [
        '"""Auto-generated Pydantic template — produced by generate_template().',
        "Re-generate it whenever the example documents change; do not hand-edit.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from pydantic import BaseModel, Field",
        "",
        "",
    ]
    for type_name, spec in sorted(template["entity_types"].items()):
        cls = f"{class_prefix}{_safe_ident(type_name)}"
        lines.append(f"class {cls}(BaseModel):")
        lines.append(f'    type: str = Field(default="{type_name}")')
        lines.append("    name: str")
        used: set[str] = set()
        for attr in spec["attributes"]:
            lines.append(f"    {_safe_ident(attr, used)}: str = ''")
        lines.append("")
    lines.append(f"class {class_prefix}Relationship(BaseModel):")
    lines.append("    source: str")
    lines.append("    target: str")
    allowed = sorted(template["relationship_types"])
    lines.append(f"    type: str  # one of: {', '.join(allowed) or 'none found'}")
    lines.append("")
    return "\n".join(lines)


def validate_template_code(code: str) -> bool:
    """The generated code must compile and define at least one model.

    Returns False (rather than raising) for code that does not compile
    or fails to execute, so callers can treat it as a validation verdict.
    """
    namespace: dict = {}
    try:
        compile(code, "<auto-template>", "exec")
        exec(code, namespace)  # noqa: S102 - generated from our own template only
    except Exception:  # noqa: BLE001 - any failure means "not valid"
        return False
    return any(isinstance(v, type) and issubclass(v, _BaseModel)
               for v in namespace.values())


try:
    from pydantic import BaseModel as _BaseModel
except ImportError:  # pragma: no cover
    _BaseModel = object  # type: ignore
