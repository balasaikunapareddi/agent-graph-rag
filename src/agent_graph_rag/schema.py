"""Validated Pydantic schema for the knowledge graph.

Every node and edge carries provenance back to the exact chunk and page it
came from. Nothing enters the graph without passing this schema — that is
the "validated Pydantic objects" half of the Docling Graph idea.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class EntityType(str, Enum):
    COMPOUND = "Compound"
    REACTION = "Reaction"
    CATALYST = "Catalyst"
    ENZYME = "Enzyme"
    PROPERTY = "Property"


class RelationshipType(str, Enum):
    PRODUCES = "PRODUCES"
    CATALYZED_BY = "CATALYZED_BY"
    HAS_PROPERTY = "HAS_PROPERTY"
    TESTED_AGAINST = "TESTED_AGAINST"


class Provenance(BaseModel):
    """Where a fact came from: document, chunk, page, and the source span."""

    doc_id: str = Field(..., description="Document id, e.g. lab-note-01-synthesis")
    chunk_id: str = Field(..., description="Chunk id, e.g. lab-note-01-synthesis:p1")
    page: int = Field(..., ge=1, description="1-based page number in the document")
    text_span: str = Field(default="", description="Verbatim source sentence(s)")


class Entity(BaseModel):
    name: str = Field(..., min_length=1, description="Canonical id as written, e.g. AZT-101")
    type: EntityType
    attributes: dict[str, str] = Field(default_factory=dict)
    provenance: Provenance

    @model_validator(mode="after")
    def _property_needs_kind(self) -> "Entity":
        if self.type == EntityType.PROPERTY and "kind" not in self.attributes:
            raise ValueError("Property entities must carry an attributes['kind']")
        return self


class Relationship(BaseModel):
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    type: RelationshipType
    provenance: Provenance

    @model_validator(mode="after")
    def _no_self_loop(self) -> "Relationship":
        if self.source == self.target:
            raise ValueError("Relationship cannot link an entity to itself")
        return self


class GraphExtraction(BaseModel):
    """One chunk's extraction: entities + relationships, all validated."""

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)

    @model_validator(mode="after")
    def _endpoints_must_be_entities(self) -> "GraphExtraction":
        names = {e.name for e in self.entities}
        for rel in self.relationships:
            if rel.source not in names or rel.target not in names:
                raise ValueError(
                    f"Relationship {rel.source} -{rel.type}-> {rel.target} "
                    "references an entity not in this extraction"
                )
        return self


# Allowed (source type, relationship, target type) triples. Anything else is
# a schema violation and is rejected at graph-build time.
ALLOWED_TRIPLES: set[tuple[EntityType, RelationshipType, EntityType]] = {
    (EntityType.REACTION, RelationshipType.PRODUCES, EntityType.COMPOUND),
    (EntityType.REACTION, RelationshipType.CATALYZED_BY, EntityType.CATALYST),
    (EntityType.COMPOUND, RelationshipType.HAS_PROPERTY, EntityType.PROPERTY),
    (EntityType.COMPOUND, RelationshipType.TESTED_AGAINST, EntityType.ENZYME),
}
