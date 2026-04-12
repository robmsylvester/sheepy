"""KNN and Centroid classifiers built on vector indices."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from sheepy.common.logger import get_logger
from sheepy.vector.embedding_classifier import EmbeddingClassifier
from sheepy.vector.vector_index import VectorIndex

logger = get_logger(__name__)


class KNNClassifier:
    """K-nearest-neighbor text classifier.

    Args:
        embedder: ``EmbeddingClassifier`` for vectorizing text.
        index: ``VectorIndex`` for nearest-neighbor lookup.
        k: Number of neighbors for voting.
    """

    def __init__(
        self,
        embedder: EmbeddingClassifier,
        index: VectorIndex,
        k: int = 5,
    ) -> None:
        """Store embedder, vector index, and neighbor count *k* for voting."""
        self.embedder = embedder
        self.index = index
        self.k = k

    def fit(self, texts: list[str], labels: list[Any]) -> None:
        """Index all training texts with their labels."""
        embeddings = self.embedder.embed(texts)
        self.index.add(embeddings, labels)
        logger.info("knn_fit", indexed=len(texts))

    def predict(self, texts: list[str]) -> list[Any]:
        """Classify texts by majority vote among k neighbors."""
        embeddings = self.embedder.embed(texts)
        neighbor_labels, _ = self.index.search(embeddings, k=self.k)
        return [Counter(neighbors).most_common(1)[0][0] for neighbors in neighbor_labels]

    def predict_proba(self, texts: list[str]) -> list[dict[Any, float]]:
        """Return per-class vote fractions for each input text."""
        embeddings = self.embedder.embed(texts)
        neighbor_labels, _ = self.index.search(embeddings, k=self.k)
        results = []
        for neighbors in neighbor_labels:
            counts = Counter(neighbors)
            total = sum(counts.values())
            results.append({label: count / total for label, count in counts.items()})
        return results


class CentroidClassifier:
    """Classify by distance to per-class centroid embeddings.

    Args:
        embedder: ``EmbeddingClassifier`` for vectorizing text.
    """

    def __init__(self, embedder: EmbeddingClassifier) -> None:
        """Keep an embedder reference; centroids are filled in ``fit``."""
        self.embedder = embedder
        self._centroids: dict[Any, np.ndarray] = {}

    def fit(self, texts: list[str], labels: list[Any]) -> None:
        """Compute per-class centroid from training embeddings."""
        embeddings = self.embedder.embed(texts)
        label_groups: dict[Any, list[np.ndarray]] = {}
        for emb, label in zip(embeddings, labels):
            label_groups.setdefault(label, []).append(emb)

        self._centroids = {
            label: np.mean(vecs, axis=0) for label, vecs in label_groups.items()
        }
        logger.info("centroid_fit", classes=list(self._centroids.keys()))

    def predict(self, texts: list[str]) -> list[Any]:
        """Classify by nearest centroid (cosine similarity)."""
        embeddings = self.embedder.embed(texts)
        labels_list = list(self._centroids.keys())
        centroid_matrix = np.stack([self._centroids[l] for l in labels_list])

        # Normalize for cosine similarity
        emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        cen_norm = centroid_matrix / (np.linalg.norm(centroid_matrix, axis=1, keepdims=True) + 1e-9)

        sims = emb_norm @ cen_norm.T
        indices = np.argmax(sims, axis=1)
        return [labels_list[i] for i in indices]
