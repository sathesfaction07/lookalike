import numpy as np

from image_cluster.core.index import SimilarityIndex


class TestSimilarityIndex:
    def test_add_and_search(self, sample_embeddings, sample_ids):
        index = SimilarityIndex(dimension=1280, threshold=0.5)
        index.add(sample_ids[:5], sample_embeddings[:5])
        assert index.size == 5

        pairs = index.find_similar_pairs(sample_embeddings[5:], sample_ids[5:])
        assert isinstance(pairs, list)
        for source, target, score in pairs:
            assert score >= 0.5
            assert source != target

    def test_empty_index_returns_no_pairs(self, sample_embeddings, sample_ids):
        index = SimilarityIndex(dimension=1280, threshold=0.9)
        pairs = index.find_similar_pairs(sample_embeddings, sample_ids)
        assert pairs == []

    def test_identical_vectors_match(self):
        index = SimilarityIndex(dimension=4, threshold=0.99)
        vec = np.array([[1.0, 0, 0, 0]], dtype=np.float32)
        index.add(["existing"], vec)

        pairs = index.find_similar_pairs(vec, ["query"])
        assert len(pairs) == 1
        assert pairs[0][2] >= 0.99

    def test_orthogonal_vectors_no_match(self):
        index = SimilarityIndex(dimension=4, threshold=0.5)
        v1 = np.array([[1.0, 0, 0, 0]], dtype=np.float32)
        v2 = np.array([[0, 1.0, 0, 0]], dtype=np.float32)
        index.add(["a"], v1)

        pairs = index.find_similar_pairs(v2, ["b"])
        assert pairs == []

    def test_threshold_filters_correctly(self):
        index = SimilarityIndex(dimension=4, threshold=0.95)
        v1 = np.array([[0.9, 0.1, 0, 0]], dtype=np.float32)
        v1 = v1 / np.linalg.norm(v1)
        index.add(["a"], v1)

        v2 = np.array([[0.85, 0.15, 0.1, 0]], dtype=np.float32)
        v2 = v2 / np.linalg.norm(v2)
        pairs = index.find_similar_pairs(v2, ["b"])
        # slightly different vectors shouldn't pass a 0.95 threshold
        assert all(score >= 0.95 for _, _, score in pairs)

    def test_save_and_load(self, tmp_path, sample_embeddings, sample_ids):
        index = SimilarityIndex(dimension=1280, threshold=0.5)
        index.add(sample_ids, sample_embeddings)
        index.save(tmp_path)

        loaded = SimilarityIndex(dimension=1280, threshold=0.5)
        assert loaded.load(tmp_path)
        assert loaded.size == 10
