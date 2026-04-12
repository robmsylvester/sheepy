"""RAG pipeline for retrieval-augmented text classification."""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable

from sheepy.common.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """Compose a retriever with a classifier for retrieval-augmented classification.

    The pipeline retrieves top-k examples from a knowledge base,
    then either does majority-vote classification or passes context
    to an LLM / model for a decision.

    Args:
        retriever: Any object with ``.retrieve(query, k)`` method.
        classify_fn: Optional callable that takes ``(query, retrieved_docs)``
            and returns a label. If ``None``, uses majority vote on retrieved labels.
        k: Number of documents to retrieve.
    """

    def __init__(
        self,
        retriever: Any,
        classify_fn: Callable[[str, list[dict]], Any] | None = None,
        k: int = 5,
    ) -> None:
        """Wire a retriever, optional custom classifier, and default top-*k*."""
        self.retriever = retriever
        self.classify_fn = classify_fn or self._majority_vote
        self.k = k

    @staticmethod
    def _majority_vote(query: str, docs: list[dict]) -> Any:
        """Pick the most common label among retrieved docs (*query* unused)."""
        labels = [d["label"] for d in docs]
        if not labels:
            return None
        return Counter(labels).most_common(1)[0][0]

    def classify(self, query: str) -> dict[str, Any]:
        """Retrieve context and classify a single query.

        Returns:
            Dict with keys ``label``, ``retrieved_docs``, and ``scores``.
        """
        docs = self.retriever.retrieve(query, k=self.k)
        label = self.classify_fn(query, docs)
        return {
            "label": label,
            "retrieved_docs": docs,
            "scores": [d.get("score", 0) for d in docs],
        }

    def classify_batch(self, queries: list[str]) -> list[dict[str, Any]]:
        """Classify a batch of queries."""
        return [self.classify(q) for q in queries]
