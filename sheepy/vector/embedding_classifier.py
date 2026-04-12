"""Embedding-based classifier using sentence-transformers or custom embed functions."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from sheepy.common.logger import get_logger

logger = get_logger(__name__)


class EmbeddingClassifier:
    """Classify text by computing embeddings and comparing against labeled reference vectors.

    Args:
        embed_fn: Callable that maps a list of strings to a numpy array ``(n, dim)``.
            If ``None``, uses sentence-transformers with *model_name*.
        model_name: sentence-transformers model name (used only when *embed_fn* is ``None``).
    """

    def __init__(
        self,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Bind a custom embedder or load sentence-transformers *model_name*."""
        if embed_fn is not None:
            self._embed = embed_fn
        else:
            self._model = self._load_sentence_transformer(model_name)
            self._embed = self._st_embed

    @staticmethod
    def _load_sentence_transformer(name: str):
        """Import and construct ``SentenceTransformer`` for *name*."""
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(name)

    def _st_embed(self, texts: list[str]) -> np.ndarray:
        """Encode *texts* with the loaded sentence-transformer model."""
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Compute embeddings for a list of texts."""
        return self._embed(texts)

    def embed_single(self, text: str) -> np.ndarray:
        """Compute embedding for a single text."""
        return self.embed([text])[0]
