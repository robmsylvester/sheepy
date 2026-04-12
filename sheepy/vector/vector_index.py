"""Unified interface over FAISS, ChromaDB, and HNSWlib vector indices."""
from __future__ import annotations

from typing import Any, Literal, Protocol

import numpy as np

from sheepy.common.logger import get_logger

logger = get_logger(__name__)


class VectorIndex:
    """Wraps a vector index backend (FAISS, ChromaDB, HNSW) behind a common API.

    Args:
        backend: One of ``"faiss"``, ``"chromadb"``, or ``"hnsw"``.
        dim: Embedding dimensionality.
        metric: Distance metric — ``"cosine"`` or ``"l2"``.
    """

    def __init__(
        self,
        backend: Literal["faiss", "chromadb", "hnsw"] = "faiss",
        dim: int = 768,
        metric: Literal["cosine", "l2"] = "cosine",
        **kwargs: Any,
    ) -> None:
        """Select vector backend, dimension, metric, and construct the native index."""
        self.backend_name = backend
        self.dim = dim
        self.metric = metric
        self._index: Any = None
        self._labels: list[Any] = []
        self._backend_kwargs = kwargs
        self._build_index()

    def _build_index(self) -> None:
        """Dispatch to FAISS, HNSW, or Chroma construction."""
        if self.backend_name == "faiss":
            self._build_faiss()
        elif self.backend_name == "hnsw":
            self._build_hnsw()
        elif self.backend_name == "chromadb":
            self._build_chromadb()
        else:
            raise ValueError(f"Unsupported backend: {self.backend_name}")

    def _build_faiss(self) -> None:
        """Create a flat inner-product or L2 FAISS index for *dim*."""
        import faiss

        if self.metric == "cosine":
            self._index = faiss.IndexFlatIP(self.dim)
        else:
            self._index = faiss.IndexFlatL2(self.dim)

    def _build_hnsw(self) -> None:
        """Initialize an HNSW graph index with default construction/search params."""
        import hnswlib

        space = "cosine" if self.metric == "cosine" else "l2"
        self._index = hnswlib.Index(space=space, dim=self.dim)
        max_elements = self._backend_kwargs.get("max_elements", 100_000)
        self._index.init_index(max_elements=max_elements, ef_construction=200, M=16)
        self._index.set_ef(50)

    def _build_chromadb(self) -> None:
        """Create an in-memory Chroma client and metadata-tagged collection."""
        import chromadb

        self._client = chromadb.Client()
        collection_name = self._backend_kwargs.get("collection", "sheepy_vectors")
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": self.metric},
        )

    def add(self, vectors: np.ndarray, labels: list[Any]) -> None:
        """Add vectors with associated labels to the index.

        Args:
            vectors: Array of shape ``(n, dim)``.
            labels: List of labels of length ``n``.
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.metric == "cosine" and self.backend_name == "faiss":
            import faiss
            faiss.normalize_L2(vectors)

        if self.backend_name == "faiss":
            self._index.add(vectors)
            self._labels.extend(labels)
        elif self.backend_name == "hnsw":
            ids = list(range(len(self._labels), len(self._labels) + len(labels)))
            self._index.add_items(vectors, ids)
            self._labels.extend(labels)
        elif self.backend_name == "chromadb":
            ids = [str(len(self._labels) + i) for i in range(len(labels))]
            self._collection.add(
                embeddings=vectors.tolist(),
                ids=ids,
                metadatas=[{"label": str(lbl)} for lbl in labels],
            )
            self._labels.extend(labels)

    def search(self, query: np.ndarray, k: int = 5) -> tuple[list[list[Any]], np.ndarray]:
        """Search for the k nearest neighbors.

        Args:
            query: Array of shape ``(n_queries, dim)``.
            k: Number of neighbors to retrieve.

        Returns:
            Tuple of (labels per query, distances per query).
        """
        query = np.ascontiguousarray(query, dtype=np.float32)

        if self.backend_name == "faiss":
            if self.metric == "cosine":
                import faiss
                faiss.normalize_L2(query)
            distances, indices = self._index.search(query, k)
            result_labels = [[self._labels[i] for i in row if i >= 0] for row in indices]
            return result_labels, distances

        elif self.backend_name == "hnsw":
            indices, distances = self._index.knn_query(query, k=k)
            result_labels = [[self._labels[i] for i in row] for row in indices]
            return result_labels, distances

        elif self.backend_name == "chromadb":
            results = self._collection.query(query_embeddings=query.tolist(), n_results=k)
            result_labels = []
            distances_out = []
            for metas, dists in zip(results["metadatas"], results["distances"]):
                result_labels.append([m["label"] for m in metas])
                distances_out.append(dists)
            return result_labels, np.array(distances_out)

    @property
    def size(self) -> int:
        """Number of vectors currently stored (labels tracked in parallel)."""
        return len(self._labels)
