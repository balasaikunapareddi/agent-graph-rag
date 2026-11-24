"""Live test: real Gemini extraction + multi-hop answers with provenance.

Run with the stored key (never printed or logged) via:
    python3 ~/workspace/custom-connectors/custom.gemini/bin/run_live.py \\
        <repo-dir> <venv-python> scripts/live_test.py

Extracts entities/relationships from all five lab notes with the LLM, builds
the graph, and answers three multi-hop questions. Prints per-question
PASS/FAIL and overall accuracy. Exit code 0 only if extraction produced
a usable graph AND at least 2 of 3 questions are answered correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_graph_rag.agent import GraphRAGAgent, default_data_dir  # noqa: E402
from agent_graph_rag.chunking import load_chunks  # noqa: E402
from agent_graph_rag.extractor import LLMExtractor  # noqa: E402

QUESTIONS = [
    {"question": "Which compound was produced by reaction R-14, and what "
                 "solubility was measured for it?",
     "expected": ["AZT-101", "12.4"]},
    {"question": "What catalyst was used in the reaction that produced "
                 "BRX-207, and what was the yield?",
     "expected": ["HATU", "64"]},
    {"question": "Which compound had the lowest IC50 against Kinase-7, "
                 "and which reaction produced it?",
     "expected": ["CLT-558", "18", "R-31"]},
]


def main() -> int:
    extractor = LLMExtractor()
    print(f"Live test — model: {extractor.model}")
    agent = GraphRAGAgent(extractor=extractor)
    chunks = [c for c in load_chunks(default_data_dir())]
    stats = agent.ingest_chunks(chunks)
    print(f"Ingested {stats['chunks']} chunks -> {stats['nodes']} nodes, "
          f"{stats['edges']} edges ({stats['relationships']} relationships)")

    # Sanity: the LLM must have found the headline facts, not an empty graph.
    graph_ok = stats["nodes"] >= 10 and stats["relationships"] >= 5
    print(f"Graph sanity (>=10 nodes, >=5 relationships): "
          f"{'PASS' if graph_ok else 'FAIL'}")

    correct = 0
    for i, item in enumerate(QUESTIONS, 1):
        answer = agent.answer(item["question"])
        ok = all(e in answer.text for e in item["expected"])
        correct += int(ok)
        print(f"\nQ{i}: {item['question']}")
        print(f"  Answer: {answer.text}")
        for line in answer.provenance_lines():
            print(f"  provenance: {line}")
        print(f"  Expected substrings: {item['expected']} -> "
              f"{'PASS' if ok else 'FAIL'}")

    print(f"\nAccuracy: {correct}/{len(QUESTIONS)} "
          f"({correct / len(QUESTIONS):.0%})")
    passed = graph_ok and correct >= 2
    print("LIVE TEST:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
