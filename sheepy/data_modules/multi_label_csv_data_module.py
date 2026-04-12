"""CSV-based data module for multilabel text classification."""
from __future__ import annotations

import os
import pickle
from typing import Any

import pandas as pd

from sheepy.common.df_ops import resample_multilabel_positives
from sheepy.common.logger import get_logger
from sheepy.data_modules.base_csv_data_module import BaseCSVDataModule
from sheepy.nlp.label_encoder import LabelEncoder

logger = get_logger(__name__)


class MultiLabelCSVDataModule(BaseCSVDataModule):
    """CSV data module for experiments with multiple binary label columns."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize CSV multilabel module, ``pos_weights`` map, and multilabel checks."""
        super().__init__(config)
        if self.text_col is None:
            raise ValueError("MultiLabelCSVDataModule requires a 'text' key in config.")
        self._verify_multilabel()
        self.pos_weights: dict[str, float] = {}

    def _verify_multilabel(self) -> None:
        """Assert ``label`` list length matches ``num_labels``."""
        labels = self.config.get("label")
        assert isinstance(labels, list), "hparams.label must be a list."
        assert len(labels) > 1, "Need more than one label for multilabel."
        assert self.config.get("num_labels") == len(labels)

    def _resample_positive_rows(self, df: pd.DataFrame, positive_label: Any = "1") -> pd.DataFrame:
        """Upsample positives per label using dict or uniform int rates."""
        rate = self.config.get("positive_resample_rate", 1)
        if isinstance(rate, dict):
            logger.info("resampling_multilabel_positives")
            return resample_multilabel_positives(df, rate, positive_label)
        elif isinstance(rate, int) and rate > 1:
            rate_dict = {k: rate for k in self.label_col}
            return resample_multilabel_positives(df, rate_dict, positive_label)
        return df

    def _load_text_from_text_file(self, filepath: str) -> list[dict]:
        """One row per line with all label columns set to None for inference."""
        prepared = []
        with open(filepath, "r") as f:
            for idx, line in enumerate(f):
                entry = {k: None for k in self.label_col}
                entry[self.text_col] = line.rstrip()
                entry[self.sample_id_col] = idx
                prepared.append(entry)
        return prepared

    def _load_text_from_raw_input(self) -> list[dict]:
        """Interactive stdin sample with empty label columns."""
        text = input("Enter sample text (Ctrl+C to exit):\n")
        entry = {self.text_col: text, self.sample_id_col: 0}
        for col in self.label_col:
            entry[col] = None
        return [entry]

    def save_data_module(self, out_path: str | None = None) -> None:
        """Pickle class sizes, label encoder, and BCE ``pos_weights``."""
        if out_path is None:
            out_path = os.path.join(self.config.get("output_dir", "."), "data.module")
        with open(out_path, "wb") as fp:
            pickle.dump({
                "class_sizes": self.train_class_sizes,
                "label_encoder": self.label_encoder,
                "pos_weights": self.pos_weights,
            }, fp)

    def load_data_module(self, in_path: str | None = None) -> None:
        """Restore encoder, class sizes, and ``pos_weights`` from disk."""
        if in_path is None:
            in_path = os.path.join(self.config.get("output_dir", "."), "data.module")
        with open(in_path, "rb") as fp:
            d = pickle.load(fp)
            self.train_class_sizes = d["class_sizes"]
            self.label_encoder = d["label_encoder"]
            self.pos_weights = d["pos_weights"]
        assert self.label_encoder.vocab_size == self.config.get("num_labels")

    def _set_class_sizes(self, positive_label: str = "1", negative_label: str = "0") -> None:
        """Per-label positive counts and neg/pos ratios for ``BCEWithLogitsLoss`` weighting."""
        self.train_class_sizes = {}
        self.pos_weights = {}
        for col in self.label_col:
            if self._train_dataset[col].nunique() != 2:
                raise ValueError(f"Label '{col}' must be binary.")
            pos = self._train_dataset[self._train_dataset[col] == positive_label].shape[0]
            neg = self._train_dataset[self._train_dataset[col] == negative_label].shape[0]
            self.train_class_sizes[col] = pos
            self.pos_weights[col] = float(neg) / pos if pos > 0 else 1.0
        logger.info("multilabel_class_sizes", sizes=self.train_class_sizes, weights=self.pos_weights)

    def _build_label_encoder(self) -> None:
        """Build a multilabel ``LabelEncoder`` from binary columns on the train split."""
        self.label_encoder = LabelEncoder.initializeFromMultilabelDataframe(
            self._train_dataset, self.config["label"]
        )
        logger.info("label_encoder_built", vocab=self.label_encoder.vocab)
