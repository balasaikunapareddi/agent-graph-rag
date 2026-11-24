"""One-command demo: ``python -m agent_graph_rag demo``.

Builds the graph from the shipped lab notes, runs the same five
multi-hop questions through GraphRAG and through the plain chunk-RAG
baseline, and prints both answers with provenance side by side.

Default extractor is the deterministic stub, so the demo runs with no API
key. ``--live`` switches to the real LLM extractor (needs GEMINI_API_KEY
or another LiteLLM key) and ``--llm-answer`` lets the LLM rephrase the
traversed answers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import DEMO_QUESTIONS, GraphRAGAgent, default_data_dir
from .exporters import export_csv, write_cypher
from .extractor import LLMExtractor, StubExtractor


def run_demo(live: bool = False, llm_answer: bool = False,
             export_dir: str | Path | None = None) -> int:
    extractor = LLMExtractor() if live else StubExtractor()
    agent = GraphRAGAgent(extractor=extractor)
    stats = agent.ingest(default_data_dir())
    print("agent-graph-rag — demo")
    print(f"Extractor: {'LLM (' + extractor.model + ')' if live else 'stub (deterministic, no key)'}")
    print(f"Ingested {stats['chunks']} chunks -> {stats['nodes']} nodes, "
          f"{stats['edges']} edges ({stats['relationships']} relationships)\n")

    comparisons = agent.run_demo_questions(use_llm=llm_answer)
    graph_ok = sum(c.graph_correct for c in comparisons)
    base_ok = sum(c.baseline_correct for c in comparisons)

    for i, comp in enumerate(comparisons, 1):
        print(f"Q{i}: {comp.question}")
        print(f"  GraphRAG {'CORRECT' if comp.graph_correct else 'WRONG'}: "
              f"{comp.graph_answer.text}")
        for line in comp.graph_answer.provenance_lines():
            print(f"    provenance: {line}")
        print(f"  Baseline {'CORRECT' if comp.baseline_correct else 'PARTIAL/WRONG'}: "
              f"{comp.baseline_answer.text}")
        print(f"    retrieved chunks: {', '.join(comp.baseline_answer.retrieved_chunk_ids)}")
        print()

    print(f"Score — GraphRAG: {graph_ok}/{len(comparisons)}  "
          f"Baseline: {base_ok}/{len(comparisons)}")
    print("Why the baseline loses: each multi-hop answer needs facts from "
          "two different notes (reaction -> compound in one, the compound's "
          "property in another). The graph traverses the link; the chunk "
          "retriever can only report what one chunk says.")

    if export_dir:
        paths = export_csv(agent.graph, Path(export_dir))
        cypher = write_cypher(agent.graph, Path(export_dir) / "graph.cypher")
        print(f"\nExports: {paths['nodes']}, {paths['edges']}, {cypher}")
    return 0 if graph_ok == len(comparisons) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-graph-rag")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run the GraphRAG vs baseline demo")
    demo.add_argument("--live", action="store_true",
                      help="Use the real LLM extractor (needs an API key)")
    demo.add_argument("--llm-answer", action="store_true",
                      help="Let the LLM rephrase traversed answers")
    demo.add_argument("--export-dir", default=None,
                      help="Also export CSV + Cypher to this directory")
    args = parser.parse_args(argv)
    if args.command == "demo":
        return run_demo(live=args.live, llm_answer=args.llm_answer,
                        export_dir=args.export_dir)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
