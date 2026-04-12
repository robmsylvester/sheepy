"""Base data module mixin for multilabel experiments."""
from __future__ import annotations

from typing import Any

from sheepy.data_modules.base_data_module import BaseDataModule


class MultiLabelBaseDataModule(BaseDataModule):
    """Validates that ``label`` config is a list of binary label columns."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Validate multilabel config (list of labels, ``num_labels``) then delegate to base."""
        super().__init__(config)
        self._verify_multilabel()

    def _verify_multilabel(self) -> None:
        """Assert ``label`` is a multi-column list aligned with ``num_labels``."""
        labels = self.config.get("label")
        assert isinstance(labels, list), "hparams.label must be a list for multilabel."
        assert len(labels) > 1, "Multilabel requires more than one label column."
        assert self.config.get("num_labels") == len(labels), (
            f"num_labels={self.config.get('num_labels')} does not match label list length {len(labels)}."
        )
