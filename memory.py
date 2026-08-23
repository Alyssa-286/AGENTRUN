"""
memory.py — long-term memory via embeddings + cosine similarity search.

This is a hand-built vector memory store, so the mechanics are visible:
text becomes a vector, vectors are compared with cosine similarity, and the
most relevant past entries are injected back into the agent context.

For Phase 2 we keep the embedder local and deterministic so the demo works
without any extra API dependency. The important part is the vector math and
retrieval flow, which is the same idea used by larger embedding systems.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def embed_text(text: str, dimension: int = 256) -> list[float]:
    """Turn text into a deterministic embedding vector."""

    vector = [0.0] * dimension
    tokens = _tokenize(text)

    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = 1.5 if token.isdigit() else 1.0
        vector[index] += sign * weight

    for left, right in zip(tokens, tokens[1:]):
        bigram = f"{left}_{right}"
        digest = hashlib.sha256(bigram.encode("utf-8"), usedforsecurity=False).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += 0.5 * sign

    return vector


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class VectorMemory:
    """Tiny JSON-backed vector store for Phase 2."""

    def __init__(self, filepath: str = "memory_store.json", dimension: int = 256):
        self.filepath = filepath
        self.dimension = dimension
        self.entries: list[dict[str, object]] = []
        self._load()

    def add(self, text: str) -> None:
        embedding = embed_text(text, dimension=self.dimension)
        self.entries.append(
            {
                "text": text,
                "embedding": embedding,
                "timestamp": time.time(),
            }
        )
        self._save()

    def search(self, query: str, top_k: int = 3, min_similarity: float = 0.2) -> list[str]:
        if not self.entries:
            return []

        query_embedding = embed_text(query, dimension=self.dimension)
        scored = [
            (cosine_similarity(query_embedding, entry["embedding"]), entry["text"])
            for entry in self.entries
            if isinstance(entry.get("embedding"), list) and isinstance(entry.get("text"), str)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        return [text for score, text in scored[:top_k] if score >= min_similarity]

    def _save(self) -> None:
        with open(self.filepath, "w", encoding="utf-8") as handle:
            json.dump(self.entries, handle, indent=2)

    def _load(self) -> None:
        if not os.path.exists(self.filepath):
            return

        try:
            with open(self.filepath, "r", encoding="utf-8") as handle:
                loaded_entries = json.load(handle)
        except (OSError, json.JSONDecodeError):
            loaded_entries = []

        if isinstance(loaded_entries, list):
            self.entries = loaded_entries

    def __len__(self) -> int:
        return len(self.entries)