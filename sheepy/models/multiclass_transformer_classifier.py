"""Backward-compatible alias for multiclass classification.

The ``MulticlassTransformerClassifier`` is now simply a ``TransformerClassifier``
with ``task="multiclass"`` set automatically.
"""
from __future__ import annotations

from typing import Any

import lightning as L

from sheepy.models.base_transformer_classifier import TransformerClassifier


class MulticlassTransformerClassifier(TransformerClassifier):
    def __init__(
        self,
        config: dict[str, Any],
        data: L.LightningDataModule,
        **kwargs: Any,
    ) -> None:
        """Backward-compatible alias forcing ``task='multiclass'``."""
        super().__init__(config, data, task="multiclass", **kwargs)
