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


    def test_metadata_upsert_is_durable_and_independent_from_embeddings(self):
        payload = {
            "path": "Artist/Album/song.flac",
            "title": "Blue Train",
            "artist": "John Coltrane",
            "album": "Blue Train",
            "albumArtist": "John Coltrane",
            "genre": "Jazz",
            "year": 1957,
            "trackNumber": 1,
            "discNumber": 1,
            "durationSeconds": 640.5,
            "metadataHash": "tag-sha256",
        }
        self.store.upsert(make_result(), MODEL_ID, 1)
        result = self.store.upsert_metadata(payload)
        self.assertEqual(result["path"], payload["path"])
        self.assertEqual(self.store.metadata_count(), 1)
        self.assertEqual(self.store.count(MODEL_ID, 1), 1)

        restarted = EmbeddingStore(self.database)
        restarted.initialize(MODEL_ID, 1)
        matches = restarted.search_metadata("Blue Train")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["title"], "Blue Train")
        self.assertEqual(matches[0]["artist"], "John Coltrane")
        self.assertEqual(matches[0]["year"], 1957)
        self.assertEqual(restarted.count(MODEL_ID, 1), 1)

    def test_metadata_search_supports_unicode_casefold_and_prefix_matching(self):
        self.store.upsert_metadata({
            "path": "Björk/Post/01.flac",
            "title": "Jóga",
            "artist": "Björk",
            "album": "Post",
            "albumArtist": "Björk",
            "genre": "Electronic",
            "year": 1995,
            "trackNumber": 1,
            "discNumber": 1,
            "durationSeconds": 300.0,
            "metadataHash": "unicode-tags",
        })
        matches = self.store.search_metadata("björ")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["artist"], "Björk")
        self.assertEqual(self.store.search_metadata("JÓG")[0]["title"], "Jóga")

    def test_metadata_search_prioritizes_exact_matches_and_is_deterministic(self):
        self.store.upsert_metadata({
            "path": "B/Album/track.flac", "title": "Blue", "artist": "Artist B"
        })
        self.store.upsert_metadata({
            "path": "A/Album/track.flac", "title": "Blue Train", "artist": "Artist A"
        })
        matches = self.store.search_metadata("Blue")
        self.assertEqual([item["path"] for item in matches],
                         ["B/Album/track.flac", "A/Album/track.flac"])

    def test_metadata_upsert_preserves_null_tag_values(self):
        self.store.upsert_metadata({"path": "Unknown/track.flac"})
        match = self.store.search_metadata("Unknown")
        self.assertEqual(match, [])
        self.assertEqual(self.store.metadata_count(), 1)
        with self.store._connect() as connection:
            row = connection.execute(
                "SELECT title, artist, year, metadata_hash FROM library_metadata WHERE path=?",
                ("Unknown/track.flac",),
            ).fetchone()
        self.assertEqual(row, (None, None, None, None))

    def test_metadata_upsert_is_idempotent_by_path_and_replaces_values(self):
        self.store.upsert_metadata({"path": "Artist/Album/track.flac", "title": "Old"})
        self.store.upsert_metadata({"path": "Artist/Album/track.flac", "title": "New"})
        self.assertEqual(self.store.metadata_count(), 1)
        self.assertEqual(self.store.search_metadata("New")[0]["title"], "New")
        self.assertEqual(self.store.search_metadata("Old"), [])

    def test_metadata_delete_is_idempotent_and_does_not_delete_embedding(self):
        self.store.upsert(make_result(), MODEL_ID, 1)
        self.store.upsert_metadata({"path": "Artist/Album/track.flac", "title": "Track"})
        first = self.store.delete_metadata("Artist/Album/track.flac")
        second = self.store.delete_metadata("Artist/Album/track.flac")
        self.assertTrue(first["deleted"])
        self.assertFalse(second["deleted"])
        self.assertEqual(self.store.metadata_count(), 0)
        self.assertEqual(self.store.count(MODEL_ID, 1), 1)

    def test_metadata_delete_normalizes_windows_separators(self):
        self.store.upsert_metadata({"path": "Artist\\\\Album\\\\track.flac", "title": "Track"})
        result = self.store.delete_metadata("Artist/Album/track.flac")
        self.assertTrue(result["deleted"])

    def test_metadata_validation_rejects_path_traversal_and_invalid_values(self):
        for path in ("../outside.flac", "/outside.flac", "C:/outside.flac", "Artist//track.flac"):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    self.store.upsert_metadata({"path": path})
        with self.assertRaisesRegex(ValueError, "year"):
            self.store.upsert_metadata({"path": "Artist/track.flac", "year": -1})
        with self.assertRaisesRegex(ValueError, "durationSeconds"):
            self.store.upsert_metadata({"path": "Artist/track.flac", "durationSeconds": float("nan")})
        with self.assertRaisesRegex(ValueError, "title"):
            self.store.upsert_metadata({"path": "Artist/track.flac", "title": 123})

    def test_metadata_search_validates_query_and_limit(self):
        with self.assertRaises(ValueError):
            self.store.search_metadata("")
        with self.assertRaises(ValueError):
            self.store.search_metadata("track", 0)
        with self.assertRaises(ValueError):
            self.store.search_metadata("track", 51)


if __name__ == "__main__":
    unittest.main()
