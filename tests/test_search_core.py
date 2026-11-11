import unittest

import numpy as np

from search_core import rank_vectors


class RankVectorsTests(unittest.TestCase):
    def test_returns_highest_cosine_scores_first(self):
        vectors = np.asarray([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.6],
        ], dtype=np.float32)

        result = rank_vectors(vectors, np.asarray([1.0, 0.0]), limit=2)

        self.assertEqual([index for index, _ in result], [0, 2])
        self.assertAlmostEqual(result[0][1], 1.0, places=6)
        self.assertAlmostEqual(result[1][1], 0.8, places=6)

    def test_normalizes_query_embedding(self):
        result = rank_vectors(np.asarray([[1.0, 0.0]]), np.asarray([5.0, 0.0]))
        self.assertAlmostEqual(result[0][1], 1.0, places=6)

    def test_rejects_zero_query_embedding(self):
        with self.assertRaisesRegex(ValueError, "empty vector"):
            rank_vectors(np.ones((2, 3)), np.zeros(3))

    def test_rejects_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "dimension"):
            rank_vectors(np.ones((2, 3)), np.ones(4))

    def test_rejects_non_matrix_index(self):
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            rank_vectors(np.ones(3), np.ones(3))

    def test_rejects_invalid_limit(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            rank_vectors(np.ones((2, 3)), np.ones(3), limit=0)

    def test_empty_index_returns_no_results(self):
        self.assertEqual(rank_vectors(np.empty((0, 3)), np.ones(3)), [])


if __name__ == "__main__":
    unittest.main()
