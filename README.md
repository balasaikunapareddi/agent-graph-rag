# agent-graph-rag

**GraphRAG agent in the spirit of Docling Graph: documents become validated Pydantic entities in a directed knowledge graph with provenance to chunk and page, so multi-hop questions get answered by traversal — with a plain chunk-RAG baseline on the same questions to show the difference.**

## The pain

Plain chunk RAG loses the relationships between things. Ask it *"Which compound was linked to this reaction, and what properties were measured for it?"* and it retrieves a passage about the reaction *or* a passage about the compound — never both, and never the link between them. Multi-hop questions fail, not because the facts are missing from the documents, but because no structure connects them.

This project fixes that the Docling Graph way: documents are converted into **validated Pydantic objects** (typed entities + directed relationships), assembled into a **directed knowledge graph** with **provenance on every node and edge** back to the exact chunk and page, and questions are answered by **traversing** the graph — reaction → compound → property — instead of by retrieving passages.

## What it does

- **Extraction** — an LLM (via LiteLLM, model from `LLM_MODEL`, default `gemini/gemini-3.5-flash-lite`) extracts typed entities (`Compound`, `Reaction`, `Catalyst`, `Enzyme`, `Property`) and relationships (`PRODUCES`, `CATALYZED_BY`, `HAS_PROPERTY`, `TESTED_AGAINST`) from each chunk. A deterministic stub extractor is included so tests and the demo run with no key.
- **Validation** — every extraction passes through Pydantic: `Property` entities must carry a `kind`, relationships must reference entities in the same extraction, no self-loops, and only the four allowed (source type, relationship, target type) triples may enter the graph. Violations are rejected, not stored.
- **Provenance** — every node and edge records `doc_id`, `chunk_id`, `page`, and the verbatim source span. Every answer cites its provenance.
- **Multi-hop answering** — deterministic traversal over the graph (BFS paths, IC50 ranking across all compounds tested against an enzyme), with an optional LLM pass that may only rephrase the traversed paths, never add facts.
- **Plain chunk-RAG baseline** — keyword retrieval + single-chunk extractive answering on the *same* documents and questions, so the demo shows the difference side by side.
- **Exports** — the graph exports to CSV (`nodes.csv`, `edges.csv`) and to Cypher for Neo4j.
- **Auto-generated templates** — `templates.generate_template()` infers the entity/relationship schema from example documents and emits a Pydantic source file that compiles and validates, Docling Graph style.

## The document set

Five fictional lab/research notes (`src/agent_graph_rag/data/`, 10 pages total) about three compounds (AZT-101, BRX-207, CLT-558), three reactions (R-14, R-22, R-31), enzyme assays against Kinase-7 and Phosphatase-2, and measured solubilities, melting points, logP values, and IC50s. The facts for any multi-hop answer deliberately live in *different* notes — that is the point.

## One-command demo

```bash
python -m venv .venv && .venv/bin/pip install -e .[test]
python -m agent_graph_rag demo
```

No API key needed — the demo uses the deterministic stub extractor. It prints, for each of five multi-hop questions, the GraphRAG answer with provenance and the baseline answer side by side:

```
Q1: Which compound was produced by reaction R-14, and what solubility was measured for it?
  GraphRAG CORRECT: Reaction R-14 produced compound AZT-101, and AZT-101 has solubility 12.4 mg/mL.
    provenance: lab-note-01-synthesis, lab-note-01-synthesis:p1, p.1
    provenance: lab-note-03-properties, lab-note-03-properties:p1, p.1
  Baseline PARTIAL/WRONG: Partial answer — the top chunk ... says: Reaction R-14 was run as a Suzuki coupling ...
Score — GraphRAG: 5/5  Baseline: 0/5
```

Live variants (need `GEMINI_API_KEY`):

```bash
python -m agent_graph_rag demo --live            # real LLM extraction
python -m agent_graph_rag demo --export-dir out  # also write nodes.csv, edges.csv, graph.cypher
```

## Verified results (run firsthand, 2026-11-24)

| Check | Result |
|---|---|
| Offline tests (`pytest`) | **69 passed** (stub LLM, deterministic) |
| Live test (`scripts/live_test.py`, real Gemini `gemini-3.5-flash-lite`) | **3/3 questions correct (100%)**, graph 26 nodes / 25 relationships |
| Live demo (`demo --live`, real Gemini) | **GraphRAG 5/5, baseline 0/5** |
| Stub demo (`demo`) | **GraphRAG 5/5, baseline 0/5**, graph 25 nodes / 24 relationships |

Notes on the live runs: the first live-test run scored 2/3 — the LLM dropped the `yield_pct` attribute because the extraction prompt did not list it. Fixing the prompt (one line in `prompts.py`) made the re-run 3/3. The live demo initially hit a Gemini **503 Service Unavailable**; the extractor now retries a transient API failure once after 20 s, and the re-run after a few minutes passed 5/5. The live graph has one extra node/relationship versus the stub graph (the LLM extracted one additional attribute the stub regexes do not cover).

## Design and framework choice

No agent framework is used — deliberately. For this build the core *is* the graph and the extraction, and **networkx + LiteLLM + Pydantic** are the natural fit: networkx for the directed graph and path-finding, LiteLLM so the model swaps by environment variable, Pydantic so nothing unvalidated enters the graph. An agent framework would add indirection without adding capability here. The answering step is deterministic traversal, which is what makes the answers checkable against provenance.

```
src/agent_graph_rag/
├── prompts.py      # system prompts (separate file, never inline)
├── schema.py       # Pydantic entities, relationships, provenance, allowed triples
├── chunking.py     # documents -> page chunks with (doc, chunk, page) provenance
├── extractor.py    # LLMExtractor (LiteLLM) + StubExtractor (deterministic regex)
├── graph.py        # KnowledgeGraph: networkx DiGraph + traversal
├── agent.py        # GraphRAGAgent: ingest -> graph -> multi-hop answers + comparison
├── baseline.py     # PlainChunkRAG: keyword retrieval, single-chunk answers
├── templates.py    # auto-generate Pydantic templates from example docs
├── exporters.py    # CSV + Cypher export
├── cli.py          # `python -m agent_graph_rag demo`
└── data/           # 5 fictional lab notes (10 pages)
```

## Honest limits

- **Fictional, tidy documents.** The lab notes are clean and consistent; real documents are messier, and LLM extraction quality drops with them. The stub extractor's regexes are tuned to the shipped phrasing and would not generalise to arbitrary documents — it exists for deterministic tests and demos, not production.
- **The baseline is deliberately naive.** It is keyword retrieval with single-chunk extractive answering — the classic chunk-RAG failure mode, not a tuned embedding retriever with reranking. A stronger retriever would do better on some questions; the point stands that retrieval alone cannot *chain* facts across chunks.
- **Answering covers the demo intents.** Traversal handles property lookups via a reaction or compound, catalyst/yield lookups, and lowest/highest IC50 ranking. Arbitrary natural-language questions outside those intents return "I cannot answer that" rather than a guess — by design.
- **Cypher export is not tested against a live Neo4j.** The statements are standard `CREATE`/`MATCH ... CREATE` and load in Neo4j 5+, but no live database was run.
- **Free-tier flakiness.** Gemini free tier returns occasional 503s (seen during this build); the extractor retries once, then fails loudly instead of silently degrading.
- **No embeddings, no vector store.** Similarity is token overlap in the baseline and exact traversal in the graph. Adding embeddings would help the baseline and is a natural next step, not a claim made here.

## License

MIT — see `LICENSE`.
