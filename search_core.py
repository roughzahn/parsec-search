"""Small, dependency-light helpers used by the parsec-search HTTP service."""
from __future__ import annotations

import numpy as np


def ui_enabled(value: str | None) -> bool:
    """Interpret the UI feature flag; unset/unknown values remain enabled for dev safety."""
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def rank_vectors(vectors: np.ndarray, query_embedding: np.ndarray, limit: int = 10) -> list[tuple[int, float]]:
    """Return (row index, cosine score) pairs, highest score first."""
    matrix = np.asarray(vectors, dtype=np.float32)
    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)

    if matrix.ndim != 2:
        raise ValueError("Search vectors must be a two-dimensional matrix")
    if matrix.shape[0] == 0:
        return []
    if matrix.shape[1] != query.size:
        raise ValueError("Query embedding dimension does not match the search index")
    if limit < 1:
        raise ValueError("Result limit must be positive")

    norm = float(np.linalg.norm(query))
    if norm == 0.0:
        raise ValueError("The model returned an empty vector")
    query = query / norm

    scores = matrix @ query
    count = min(limit, scores.size)
    indices = np.argsort(scores)[-count:][::-1]
    return [(int(index), float(scores[index])) for index in indices]
