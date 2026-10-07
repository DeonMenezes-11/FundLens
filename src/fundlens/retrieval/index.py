"""Hybrid retrieval: dense embeddings (bge-small via fastembed, ONNX on CPU) + BM25, merged with
reciprocal rank fusion.

Why hybrid: regulatory text is full of exact tokens ("TER", "Rs. 500 crore", "exit load", circular numbers)
that dense embeddings blur, while paraphrased questions ("how much can a fund charge me?") need semantics.
RRF combines the two rankings without having to calibrate their very different score scales.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from ..config import Settings
from ..ingest.documents import Chunk, load_chunks

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = set("a an the of to in for on and or is are be by with as at from that this it its which shall".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank(d))."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None
    bm25_rank: int | None


class Embedder:
    def __init__(self, model: str):
        from fastembed import TextEmbedding  # imported lazily: tests don't need the ONNX model

        self._model = TextEmbedding(model_name=model)

    def embed(self, texts: list[str], query: bool = False) -> np.ndarray:
        if query:
            vecs = list(self._model.query_embed(texts))
        else:
            vecs = list(self._model.passage_embed(texts))
        arr = np.asarray(vecs, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


class HybridIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray | None, embedder: Embedder | None):
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    # -- persistence -----------------------------------------------------------------------------------
    @classmethod
    def build(cls, settings: Settings) -> "HybridIndex":
        chunks = load_chunks(settings)
        embedder = Embedder(settings.embed_model)
        vectors = embedder.embed([f"{c.title}. {c.text}" for c in chunks])
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(settings.index_dir / "vectors.npy", vectors)
        (settings.index_dir / "meta.json").write_text(
            json.dumps({"model": settings.embed_model, "n": len(chunks), "dim": int(vectors.shape[1])})
        )
        return cls(chunks, vectors, embedder)

    @classmethod
    def load(cls, settings: Settings, dense: bool = True) -> "HybridIndex":
        chunks = load_chunks(settings)
        if not dense:
            return cls(chunks, None, None)
        vectors = np.load(settings.index_dir / "vectors.npy")
        return cls(chunks, vectors, Embedder(settings.embed_model))

    def with_extra(self, extra: list[Chunk]) -> "HybridIndex":
        """A copy of this index with additional chunks (used by the eval to plant poisoned documents)."""
        vectors = self.vectors
        if vectors is not None and self.embedder is not None:
            vectors = np.vstack([vectors, self.embedder.embed([f"{c.title}. {c.text}" for c in extra])])
        return HybridIndex(self.chunks + extra, vectors, self.embedder)

    # -- search ----------------------------------------------------------------------------------------
    def search(self, query: str, k: int = 5, pool: int = 30, mode: str = "hybrid") -> list[Hit]:
        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_rank = list(np.argsort(-bm25_scores)[:pool])
        rankings = [bm25_rank] if mode in ("hybrid", "bm25") else []
        dense_rank: list[int] = []
        if mode in ("hybrid", "dense") and self.vectors is not None and self.embedder is not None:
            q = self.embedder.embed([query], query=True)[0]
            dense_rank = list(np.argsort(-(self.vectors @ q))[:pool])
            rankings.append(dense_rank)
        fused = rrf(rankings)[:k]
        pos_d = {idx: r for r, idx in enumerate(dense_rank)}
        pos_b = {idx: r for r, idx in enumerate(bm25_rank)}
        return [Hit(self.chunks[i], s, pos_d.get(i), pos_b.get(i)) for i, s in fused]
