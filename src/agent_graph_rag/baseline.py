"""Plain chunk-RAG baseline: the thing GraphRAG is compared against.

Same documents, same questions, no graph. Retrieval is token-overlap
ranking over chunks (no embeddings, no key); the answer is extracted from
the single best chunk only. That single-chunk limit is the honest point of
the comparison: a multi-hop question needs facts from two different chunks
(e.g. the reaction that produced a compound lives in one note, the
compound's solubility in another), and a retriever that returns passages
cannot chain them — it can only report what one chunk says. The baseline
therefore answers "partial" whenever the second hop is not in its top
chunk, and says so instead of guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .chunking import Chunk

STOPWORDS = set(
    "which what was the a an is are did does do of for in on to and or with "
    "that this it its by from as at be been were has have had not".split())

TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")


@dataclass
class BaselineAnswer:
    text: str
    retrieved_chunk_ids: list[str]
    complete: bool  # False = partial: a hop is missing from the top chunk


def _tokens(text: str) -> set[str]:
    return {t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS}


class PlainChunkRAG:
    """Keyword retrieval + single-chunk extractive answering."""

    def __init__(self, chunks: list[Chunk], top_k: int = 2):
        self.chunks = chunks
        self.top_k = top_k
        self._index = [(c, _tokens(c.text)) for c in chunks]

    def retrieve(self, question: str) -> list[tuple[Chunk, int]]:
        q = _tokens(question)
        scored = sorted(
            ((c, len(q & toks)) for c, toks in self._index),
            key=lambda pair: pair[1], reverse=True)
        return [(c, s) for c, s in scored[: self.top_k] if s > 0]

    def answer(self, question: str) -> BaselineAnswer:
        retrieved = self.retrieve(question)
        if not retrieved:
            return BaselineAnswer(
                "No relevant chunk found.", [], complete=False)
        top_chunk, _ = retrieved[0]
        chunk_ids = [c.chunk_id for c, _ in retrieved]
        text = top_chunk.text

        # Extractive patterns for the question intents used in the demo.
        q = question.lower()
        patterns = [
            (r"solubility", r"[Tt]he solubility of ([A-Z]{2,4}-\d+) was ([\d.]+) mg/mL"),
            (r"melting point", r"[Tt]he melting point of ([A-Z]{2,4}-\d+) was (\d+) °C"),
            (r"logp", r"[Tt]he logP of ([A-Z]{2,4}-\d+) was ([\d.]+)"),
            (r"ic50", r"([A-Z]{2,4}-\d+) was tested against enzyme ([A-Za-z]+-\d+) with an IC50 of ([\d,]+) nM"),
            (r"catalyst", r"[Tt]he catalyst was ([A-Za-z][A-Za-z0-9()]+)"),
            (r"yield", r"with a yield of (\d+)%"),
        ]
        for keyword, pattern in patterns:
            if keyword in q:
                m = re.search(pattern, text)
                if m:
                    return BaselineAnswer(
                        f"From the top retrieved chunk ({top_chunk.chunk_id}): "
                        + m.group(0), chunk_ids, complete=True)
        # No extractable fact in the top chunk: report what it does say.
        skip = ("#", "Date:", "Researcher:", "Method:", "Assay:", "Project:")
        first_line = next((ln.strip() for ln in text.splitlines()
                           if ln.strip() and not ln.startswith(skip)),
                          text[:120])
        mentions_compound = re.search(r"\b([A-Z]{2,4}-\d+)\b", question)
        if mentions_compound:
            return BaselineAnswer(
                f"Partial answer — the top chunk ({top_chunk.chunk_id}) says: "
                f"{first_line} The other hop of this question is in a "
                "different chunk and this retriever cannot chain across "
                "chunks.", chunk_ids, complete=False)
        return BaselineAnswer(
            f"Partial answer — the top chunk ({top_chunk.chunk_id}) says: "
            f"{first_line} This question needs a second hop (e.g. which "
            "compound a reaction produced) that lives in a different chunk, "
            "so the single-chunk answer is incomplete.",
            chunk_ids, complete=False)
