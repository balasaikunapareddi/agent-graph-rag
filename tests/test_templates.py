"""Template generation tests: schema inferred from example docs."""

from agent_graph_rag.agent import default_data_dir
from agent_graph_rag.chunking import load_chunks
from agent_graph_rag.extractor import StubExtractor
from agent_graph_rag.templates import (
    generate_template,
    template_to_pydantic_code,
    validate_template_code,
)


def test_template_finds_entity_types():
    template = generate_template(load_chunks(default_data_dir()), StubExtractor())
    assert set(template["entity_types"]) == {
        "Compound", "Reaction", "Catalyst", "Enzyme", "Property"}
    assert template["generated_from_chunks"] == 10


def test_template_records_property_attributes():
    template = generate_template(load_chunks(default_data_dir()), StubExtractor())
    attrs = template["entity_types"]["Property"]["attributes"]
    assert {"kind", "value", "unit"} <= set(attrs)


def test_template_records_relationship_pairs():
    template = generate_template(load_chunks(default_data_dir()), StubExtractor())
    assert set(template["relationship_types"]) == {
        "PRODUCES", "CATALYZED_BY", "HAS_PROPERTY", "TESTED_AGAINST"}
    assert "Reaction->Compound" in template["relationship_types"]["PRODUCES"]["pairs"]


def test_generated_code_compiles_and_defines_models():
    template = generate_template(load_chunks(default_data_dir()), StubExtractor())
    code = template_to_pydantic_code(template)
    assert "class AutoCompound(BaseModel)" in code
    assert "class AutoRelationship(BaseModel)" in code
    assert validate_template_code(code)


def test_generated_model_validates_an_entity():
    template = generate_template(load_chunks(default_data_dir()), StubExtractor())
    code = template_to_pydantic_code(template)
    namespace: dict = {}
    exec(code, namespace)  # noqa: S102 - generated from our own template
    compound = namespace["AutoCompound"](name="AZT-101")
    assert compound.name == "AZT-101" and compound.type == "Compound"
