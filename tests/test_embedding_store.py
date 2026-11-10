import os
import tempfile
import unittest

import numpy as np

from embedding_store import EmbeddingStore

MODEL_ID = "laion-clap-630k-audioset-v1"


def make_result(path="Artist/Album/track.flac", vector=None, size=12345):
    values = np.ones(512, dtype=np.float32) if vector is None else np.asarray(vector, dtype=np.float32)
    return {
        "modelId": MODEL_ID,
        "embeddingVersion": 1,
        "dimensions": 512,
        "embedding": values.tolist(),
        "audio": {
            "path": path,
            "sizeBytes": size,
            "lastModifiedUtc": "2026-11-09T12:00:00Z",
            "contentHash": "sha256:abc",
        },
    }


class EmbeddingStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = os.path.join(self.temp.name, "embeddings.db")
        self.store = EmbeddingStore(self.database)
        self.store.initialize(MODEL_ID, 1)

    def tearDown(self):
        self.temp.cleanup()

    def test_upsert_persists_and_refreshes_snapshot(self):
        self.store.upsert(make_result(), MODEL_ID, 1)
        vectors, paths = self.store.snapshot(MODEL_ID, 1)
        self.assertEqual(paths, ["Artist/Album/track.flac"])
        self.assertEqual(vectors.shape, (1, 512))
        self.assertAlmostEqual(float(np.linalg.norm(vectors[0])), 1.0, places=5)

        restarted = EmbeddingStore(self.database)
        restarted.initialize(MODEL_ID, 1)
        vectors_after_restart, paths_after_restart = restarted.snapshot(MODEL_ID, 1)
        self.assertEqual(paths_after_restart, paths)
        np.testing.assert_allclose(vectors_after_restart, vectors)

    def test_repeated_ingestion_is_idempotent_by_path(self):
        self.store.upsert(make_result(vector=np.ones(512)), MODEL_ID, 1)
        self.store.upsert(make_result(vector=np.full(512, 2.0), size=54321), MODEL_ID, 1)
        self.assertEqual(self.store.count(MODEL_ID, 1), 1)
        vectors, _ = self.store.snapshot(MODEL_ID, 1)
        np.testing.assert_allclose(vectors[0], np.full(512, 1 / np.sqrt(512)), rtol=1e-5)

    def test_delete_removes_matching_revision_and_refreshes_snapshot(self):
        self.store.upsert(make_result(), MODEL_ID, 1)
        result = self.store.delete(
            "Artist/Album/track.flac", 12345, "2026-11-09T12:00:00Z"
        )
        self.assertTrue(result["deleted"])
        vectors, paths = self.store.snapshot(MODEL_ID, 1)
        self.assertEqual(paths, [])
        self.assertEqual(vectors.shape, (0, 512))

    def test_delete_does_not_remove_a_newer_revision(self):
        self.store.upsert(make_result(size=54321), MODEL_ID, 1)
        result = self.store.delete(
            "Artist/Album/track.flac", 12345, "2026-11-09T12:00:00Z"
        )
        self.assertFalse(result["deleted"])
        self.assertEqual(self.store.count(MODEL_ID, 1), 1)

    def test_delete_can_remove_legacy_embedding_without_revision_metadata(self):
        with self.store._connect() as connection:
            connection.execute("""
                INSERT INTO embeddings(path, model_id, embedding_version, dimensions, vector, updated_at)
                VALUES(?, ?, 1, 512, ?, ?)
            """, ("Legacy/track.flac", MODEL_ID, np.ones(512, dtype=np.float32).tobytes(), "now"))
        result = self.store.delete("Legacy/track.flac", 100, "2026-11-09T12:00:00Z")
        self.assertTrue(result["deleted"])
        self.assertEqual(self.store.count(MODEL_ID, 1), 0)

    def test_rejects_wrong_model_and_invalid_vector(self):
        payload = make_result()
        payload["modelId"] = "other-model"
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.store.upsert(payload, MODEL_ID, 1)

        payload = make_result()
        payload["embedding"] = [0.1, 0.2]
        payload["dimensions"] = 2
        with self.assertRaisesRegex(ValueError, "512"):
            self.store.upsert(payload, MODEL_ID, 1)

    def test_rejects_path_traversal(self):
        with self.assertRaisesRegex(ValueError, "escape"):
            self.store.upsert(make_result(path="../outside.flac"), MODEL_ID, 1)

    def test_imports_legacy_npz_once(self):
        legacy_path = os.path.join(self.temp.name, "embedding-index.npz")
        np.savez_compressed(
            legacy_path,
            vectors=np.ones((2, 512), dtype=np.float32),
            paths=np.asarray(["A/one.flac", "B/two.flac"]),
            ids=np.asarray(["one", "two"]),
        )
        database = os.path.join(self.temp.name, "legacy.db")
        migrated = EmbeddingStore(database, legacy_path)
        migrated.initialize(MODEL_ID, 1)
        self.assertEqual(migrated.count(MODEL_ID, 1), 2)
        vectors, paths = migrated.snapshot(MODEL_ID, 1)
        self.assertEqual(paths, ["A/one.flac", "B/two.flac"])
        self.assertEqual(vectors.shape, (2, 512))


if __name__ == "__main__":
    unittest.main()
