"""Dense and sparse retrieval components for RAG-based classification."""
from __future__ import annotations

from typing import Any, Callable, Protocol

import numpy as np

from sheepy.common.logger import get_logger

logger = get_logger(__name__)


class Retriever(Protocol):
    """Protocol that all retrievers must satisfy."""

    def index(self, documents: list[str], labels: list[Any], **kwargs: Any) -> None:
        """Store corpus *documents* with parallel *labels* for later search."""
        ...

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return up to *k* scored hits for *query* as ``text`` / ``label`` / ``score`` dicts."""
        ...


class DenseRetriever:
    """Retrieve documents using dense embeddings.

    Args:
        embed_fn: Maps a list of strings to a numpy array ``(n, dim)``.
            If ``None``, uses sentence-transformers.
        model_name: sentence-transformers model name (fallback).
    """

    def __init__(
        self,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Use *embed_fn* or lazily wrap sentence-transformers *model_name* for encoding."""
        if embed_fn is not None:
            self._embed = embed_fn
        else:
            from sheepy.vector.embedding_classifier import EmbeddingClassifier
            self._embedder = EmbeddingClassifier(model_name=model_name)
            self._embed = self._embedder.embed

        self._documents: list[str] = []
        self._labels: list[Any] = []
        self._embeddings: np.ndarray | None = None

    def index(self, documents: list[str], labels: list[Any], **kwargs: Any) -> None:
        """Build the dense index."""
        self._documents = documents
        self._labels = labels
        self._embeddings = self._embed(documents)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-9
        self._embeddings = self._embeddings / norms
        logger.info("dense_retriever_indexed", count=len(documents))

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Retrieve top-k documents by cosine similarity."""
        q_emb = self._embed([query])[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        scores = self._embeddings @ q_emb
        top_k = np.argsort(scores)[::-1][:k]
        return [
            {"text": self._documents[i], "label": self._labels[i], "score": float(scores[i])}
            for i in top_k
        ]


class BM25Retriever:
    """Retrieve documents using BM25 sparse scoring.

    Requires ``rank-bm25`` (install with ``pip install sheepy[rag]``).
    """

    def __init__(self) -> None:
        """Initialize empty corpus; call ``index`` before ``retrieve``."""
        self._documents: list[str] = []
        self._labels: list[Any] = []
        self._bm25: Any = None

    def index(self, documents: list[str], labels: list[Any], **kwargs: Any) -> None:
        """Tokenize lowercase words and fit ``BM25Okapi`` over the corpus."""
        from rank_bm25 import BM25Okapi

        self._documents = documents
        self._labels = labels
        tokenized = [doc.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("bm25_retriever_indexed", count=len(documents))

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return top-*k* BM25-scored documents for whitespace-tokenized *query*."""
        scores = self._bm25.get_scores(query.lower().split())
        top_k = np.argsort(scores)[::-1][:k]
        return [
            {"text": self._documents[i], "label": self._labels[i], "score": float(scores[i])}
            for i in top_k
        ]
