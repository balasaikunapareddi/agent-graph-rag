"""agent-graph-rag: documents become a validated knowledge graph; multi-hop
questions are answered by traversal, with a plain chunk-RAG baseline on the
same questions to show the difference."""

from .agent import GraphRAGAgent
from .graph import KnowledgeGraph

__all__ = ["GraphRAGAgent", "KnowledgeGraph"]
__version__ = "0.1.0"
