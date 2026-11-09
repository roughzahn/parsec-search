import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import PurePosixPath

import numpy as np


class EmbeddingStore:
    """SQLite source of truth plus an immutable in-memory search snapshot."""

    def __init__(self, database_path: str, legacy_index_path: str | None = None):
        self.database_path = database_path
        self.legacy_index_path = legacy_index_path
        self._lock = threading.RLock()
        self._dirty = True
        self._snapshot = (np.empty((0, 512), dtype=np.float32), [])

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=30)
        try:
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self, default_model_id: str, default_embedding_version: int):
        os.makedirs(os.path.dirname(os.path.abspath(self.database_path)), exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    path TEXT PRIMARY KEY,
                    size_bytes INTEGER,
                    last_modified_utc TEXT,
                    content_hash TEXT,
                    model_id TEXT NOT NULL,
                    embedding_version INTEGER NOT NULL,
                    dimensions INTEGER NOT NULL CHECK (dimensions = 512),
                    vector BLOB NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS search_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            connection.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_id, embedding_version)")
            marker = connection.execute(
                "SELECT value FROM search_metadata WHERE key='legacy_npz_imported'"
            ).fetchone()
            count = connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            if marker is None and count == 0 and self.legacy_index_path and os.path.isfile(self.legacy_index_path):
                self._import_legacy(connection, default_model_id, default_embedding_version)
            connection.execute(
                "INSERT OR IGNORE INTO search_metadata(key, value) VALUES('legacy_npz_imported', 'true')"
            )
        self.refresh(default_model_id, default_embedding_version)

    def _import_legacy(self, connection, model_id: str, embedding_version: int):
        saved = np.load(self.legacy_index_path, allow_pickle=False)
        try:
            vectors = np.asarray(saved["vectors"], dtype=np.float32)
            paths = saved["paths"].tolist()
            if vectors.ndim != 2 or vectors.shape[1] != 512 or len(paths) != vectors.shape[0]:
                raise ValueError("Legacy NPZ index has an unexpected shape")
            now = datetime.now(timezone.utc).isoformat()
            rows = []
            for path, vector in zip(paths, vectors):
                if not isinstance(path, str) or not path or not np.isfinite(vector).all():
                    continue
                norm = float(np.linalg.norm(vector))
                if norm <= 0:
                    continue
                normalized = np.asarray(vector / norm, dtype=np.float32)
                rows.append((path, model_id, embedding_version, 512, normalized.tobytes(), now))
            connection.executemany("""
                INSERT OR IGNORE INTO embeddings(
                    path, model_id, embedding_version, dimensions, vector, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?)
            """, rows)
        finally:
            saved.close()

    def upsert(self, result: dict, expected_model_id: str, expected_embedding_version: int):
        if not isinstance(result, dict):
            raise ValueError("JSON object required")
        model_id = result.get("modelId")
        version = result.get("embeddingVersion")
        dimensions = result.get("dimensions")
        try:
            vector = np.asarray(result.get("embedding"), dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("Embedding must be a numeric vector") from exc
        audio = result.get("audio")
        if model_id != expected_model_id or version != expected_embedding_version:
            raise ValueError("Unsupported embedding model or version")
        if dimensions != 512 or vector.shape != (512,):
            raise ValueError("Embedding must contain exactly 512 dimensions")
        if not np.isfinite(vector).all():
            raise ValueError("Embedding values must be finite")
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("Embedding vector must not be zero")
        if not isinstance(audio, dict):
            raise ValueError("audio metadata is required")
        path = audio.get("path")
        size_bytes = audio.get("sizeBytes")
        if not isinstance(path, str) or not path.strip() or path.startswith(("/", "\\")):
            raise ValueError("audio.path must be a non-empty relative music path")
        normalized_path = path.replace("\\", "/")
        parsed_path = PurePosixPath(normalized_path)
        if parsed_path.is_absolute() or ".." in parsed_path.parts or ":" in normalized_path:
            raise ValueError("audio.path must not escape the music root")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            raise ValueError("audio.sizeBytes must be a non-negative integer")

        normalized = np.asarray(vector / norm, dtype=np.float32)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute("""
                INSERT INTO embeddings(
                    path, size_bytes, last_modified_utc, content_hash, model_id,
                    embedding_version, dimensions, vector, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, 512, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    size_bytes=excluded.size_bytes,
                    last_modified_utc=excluded.last_modified_utc,
                    content_hash=excluded.content_hash,
                    model_id=excluded.model_id,
                    embedding_version=excluded.embedding_version,
                    dimensions=excluded.dimensions,
                    vector=excluded.vector,
                    updated_at=excluded.updated_at
            """, (
                path, size_bytes, audio.get("lastModifiedUtc"), audio.get("contentHash"),
                model_id, version, normalized.tobytes(), now,
            ))
            self._dirty = True
        return {"path": path, "modelId": model_id, "embeddingVersion": version, "updatedAt": now}

    def refresh(self, model_id: str, embedding_version: int):
        with self._lock:
            if not self._dirty:
                return self._snapshot
            with self._connect() as connection:
                rows = connection.execute("""
                    SELECT path, vector FROM embeddings
                    WHERE model_id=? AND embedding_version=?
                    ORDER BY path
                """, (model_id, embedding_version)).fetchall()
            paths = [row[0] for row in rows]
            if rows:
                vectors = np.stack([
                    np.frombuffer(row[1], dtype=np.float32, count=512).copy()
                    for row in rows
                ]).astype(np.float32, copy=False)
            else:
                vectors = np.empty((0, 512), dtype=np.float32)
            self._snapshot = (vectors, paths)
            self._dirty = False
            return self._snapshot

    def snapshot(self, model_id: str, embedding_version: int):
        return self.refresh(model_id, embedding_version)

    def count(self, model_id: str, embedding_version: int) -> int:
        with self._connect() as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model_id=? AND embedding_version=?",
                (model_id, embedding_version),
            ).fetchone()[0]
