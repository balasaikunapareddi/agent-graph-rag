"""Example: auto-generate a Pydantic template from the example documents.

Usage:
    python examples/generate_template.py
"""

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_graph_rag.agent import default_data_dir
from agent_graph_rag.chunking import load_chunks
from agent_graph_rag.extractor import StubExtractor
from agent_graph_rag.templates import (
    generate_template,
    template_to_pydantic_code,
    validate_template_code,
)

template = generate_template(load_chunks(default_data_dir()), StubExtractor())
print("Entity types:", sorted(template["entity_types"]))
print("Relationship types:", sorted(template["relationship_types"]))
code = template_to_pydantic_code(template)
print("Generated code compiles and defines models:", validate_template_code(code))
print(code)
