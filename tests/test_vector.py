"""Tests for sheepy.vector module with synthetic data."""
import numpy as np
import pytest


class TestKNNClassifier:
    def _make_embedder(self):
        """Create a mock embedder that maps text to a simple hash-based vector."""
        from sheepy.vector.embedding_classifier import EmbeddingClassifier

        def simple_embed(texts):
            vecs = []
            for t in texts:
                rng = np.random.RandomState(hash(t) % 2**31)
                vecs.append(rng.randn(32).astype(np.float32))
            return np.array(vecs)

        return EmbeddingClassifier(embed_fn=simple_embed)

    def test_fit_and_predict(self):
        from sheepy.vector.knn_classifier import KNNClassifier
        from sheepy.vector.vector_index import VectorIndex

        embedder = self._make_embedder()
        index = VectorIndex(backend="faiss", dim=32, metric="l2")
        knn = KNNClassifier(embedder, index, k=3)

        texts = ["hello world", "goodbye world", "hi there", "bye there"]
        labels = ["greet", "farewell", "greet", "farewell"]
        knn.fit(texts, labels)

        assert index.size == 4
        preds = knn.predict(["hello world"])
        assert len(preds) == 1
        assert preds[0] in ("greet", "farewell")

    def test_predict_proba(self):
        from sheepy.vector.knn_classifier import KNNClassifier
        from sheepy.vector.vector_index import VectorIndex

        embedder = self._make_embedder()
        index = VectorIndex(backend="faiss", dim=32, metric="l2")
        knn = KNNClassifier(embedder, index, k=2)
        knn.fit(["a", "b"], ["x", "y"])

        probas = knn.predict_proba(["a"])
        assert len(probas) == 1
        assert isinstance(probas[0], dict)
        assert sum(probas[0].values()) == pytest.approx(1.0)


class TestCentroidClassifier:
    def test_fit_and_predict(self):
        from sheepy.vector.embedding_classifier import EmbeddingClassifier
        from sheepy.vector.knn_classifier import CentroidClassifier

        def embed(texts):
            vecs = []
            for t in texts:
                if "pos" in t:
                    vecs.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))
                else:
                    vecs.append(np.array([0.0, 1.0, 0.0], dtype=np.float32))
            return np.array(vecs)

        embedder = EmbeddingClassifier(embed_fn=embed)
        clf = CentroidClassifier(embedder)
        clf.fit(["pos_a", "pos_b", "neg_a", "neg_b"], ["positive", "positive", "negative", "negative"])

        preds = clf.predict(["pos_test"])
        assert preds == ["positive"]


class TestVectorIndex:
    def test_faiss_add_and_search(self):
        from sheepy.vector.vector_index import VectorIndex

        idx = VectorIndex(backend="faiss", dim=4, metric="l2")
        vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
        idx.add(vecs, ["a", "b", "c"])
        assert idx.size == 3

        query = np.array([[1, 0, 0, 0]], dtype=np.float32)
        labels, dists = idx.search(query, k=2)
        assert len(labels[0]) == 2
        assert labels[0][0] == "a"
