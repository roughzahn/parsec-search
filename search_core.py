"""Small, dependency-light helpers used by the parsec-search HTTP service."""
from __future__ import annotations

import numpy as np


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



def rank_similar_tracks(
    vectors: np.ndarray,
    paths: list[str],
    source_path: str,
    limit: int = 10,
) -> list[tuple[int, float]]:
    """Return nearest tracks by audio embedding, excluding the source track.

    Ties are resolved by relative path so repeated requests are deterministic.
    """
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Search vectors must be a two-dimensional matrix")
    if len(paths) != matrix.shape[0]:
        raise ValueError("Track paths do not match the embedding index")
    if not isinstance(source_path, str) or not source_path.strip():
        raise ValueError("Source track path is required")
    if limit < 1:
        raise ValueError("Result limit must be positive")

    try:
        source_index = paths.index(source_path)
    except ValueError as exc:
        raise KeyError(source_path) from exc

    if matrix.shape[1] == 0:
        raise ValueError("Search vectors must not be empty")
    scores = matrix @ matrix[source_index]
    candidates = [
        (index, float(scores[index]))
        for index in range(len(paths))
        if index != source_index
    ]
    candidates.sort(key=lambda item: (-item[1], paths[item[0]]))
    return candidates[:limit]
