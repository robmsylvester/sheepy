"""Backward-compatible alias for multilabel classification.

The ``MultiLabelTransformerClassifier`` is now simply a ``TransformerClassifier``
with ``task="multilabel"`` set automatically.
"""
from __future__ import annotations

from typing import Any

import lightning as L

from sheepy.models.base_transformer_classifier import TransformerClassifier


class MultiLabelTransformerClassifier(TransformerClassifier):
    def __init__(
        self,
        config: dict[str, Any],
        data: L.LightningDataModule,
        **kwargs: Any,
    ) -> None:
        """Backward-compatible alias forcing ``task='multilabel'``."""
        super().__init__(config, data, task="multilabel", **kwargs)
