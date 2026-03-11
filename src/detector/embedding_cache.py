"""Disk cache for response embeddings.

Stores embeddings as .npz files keyed by (response_text, model/method).
Follows the caching pattern from src/detector/sampling.py.
"""

import hashlib
from pathlib import Path

import numpy as np


CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache" / "embeddings"


def _cache_path(method_key: str) -> Path:
    """Single .npz file per embedding method."""
    return CACHE_DIR / f"{method_key}.npz"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class EmbeddingCache:
    """In-memory cache backed by a single .npz file per method.

    Usage:
        cache = EmbeddingCache("tfidf_3gram")
        emb = cache.get("some text")
        if emb is None:
            emb = compute_embedding("some text")
            cache.put("some text", emb)
        cache.save()  # persist to disk
    """

    def __init__(self, method_key: str):
        self.method_key = method_key
        self._store: dict[str, np.ndarray] = {}
        self._dirty = False
        self._load()

    def _load(self):
        path = _cache_path(self.method_key)
        if path.exists():
            data = np.load(path, allow_pickle=True)
            for key in data.files:
                self._store[key] = data[key]

    def get(self, text: str) -> np.ndarray | None:
        h = _hash_text(text)
        return self._store.get(h)

    def put(self, text: str, embedding: np.ndarray):
        h = _hash_text(text)
        self._store[h] = embedding
        self._dirty = True

    def save(self):
        if not self._dirty:
            return
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(_cache_path(self.method_key), **self._store)
        self._dirty = False

    def __len__(self):
        return len(self._store)
