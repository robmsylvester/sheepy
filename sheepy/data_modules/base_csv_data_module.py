"""CSV-based data module for single-label text classification."""
from __future__ import annotations

import os
import pickle
from typing import Any

import pandas as pd

from sheepy.common.df_ops import (
    map_labels,
    read_csv_text_classifier,
    resample_positives,
    split_dataframes,
)
from sheepy.common.logger import get_logger
from sheepy.data_modules.base_data_module import BaseDataModule

logger = get_logger(__name__)


class BaseCSVDataModule(BaseDataModule):
    """Data module that reads CSV files for text classification experiments."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize paths, optional ``x_cols``, and validate CSV data directories unless evaluating."""
        super().__init__(config)
        self.label_encoder = None
        self.train_class_sizes = None
        self._validate_data_dir_args()
        self.config.setdefault("x_cols", [])

        if self.text_col is None:
            raise ValueError("BaseCSVDataModule requires a 'text' key in hparams config.")

    def _validate_data_dir_args(self) -> None:
        """Ensure train/val/test dirs or ``data_dir`` exist; create ``data_dir`` when configured."""
        if self.evaluate:
            return
        if "data_dir" in self.config:
            data_dir = self.config["data_dir"]
            if not os.path.exists(data_dir):
                logger.info("creating_data_dir", path=data_dir)
                os.makedirs(data_dir, exist_ok=True)
        else:
            for key in ("train_data_dir", "val_data_dir", "test_data_dir"):
                path = self.config.get(key)
                if path is None or not os.path.exists(path):
                    raise ValueError(f"Directory for '{key}' does not exist: {path}")

    def _verify_module(self) -> None:
        """Compute class counts and build the label encoder after data is loaded."""
        self._set_class_sizes()
        self._build_label_encoder()

    def save_data_module(self, out_path: str | None = None) -> None:
        """Persist label encoder and training class sizes to a pickle file."""
        if out_path is None:
            out_path = os.path.join(self.config.get("output_dir", "."), "data.module")
        module_dict = {"class_sizes": self.train_class_sizes, "label_encoder": self.label_encoder}
        with open(out_path, "wb") as fp:
            pickle.dump(module_dict, fp)

    def load_data_module(self, in_path: str | None = None) -> None:
        """Restore label encoder and class sizes from disk; asserts ``num_labels`` matches."""
        if in_path is None:
            in_path = os.path.join(self.config.get("output_dir", "."), "data.module")
        with open(in_path, "rb") as fp:
            d = pickle.load(fp)
            self.train_class_sizes = d["class_sizes"]
            self.label_encoder = d["label_encoder"]
        assert self.label_encoder.vocab_size == self.config.get("num_labels")

    def _read_csv_directory(self, csv_path: str) -> list[pd.DataFrame]:
        """Load one or many CSVs from a file or directory into standardized dataframes."""
        if os.path.isdir(csv_path):
            csvs = [os.path.join(csv_path, f) for f in os.listdir(csv_path) if f.endswith(".csv")]
        elif csv_path.endswith(".csv"):
            csvs = [csv_path]
        else:
            raise ValueError(f"'{csv_path}' is neither a CSV file nor a directory.")
        if not csvs:
            raise ValueError(f"No CSV files found in '{csv_path}'.")

        return [
            read_csv_text_classifier(
                fpath,
                evaluate=self.evaluate,
                label_cols=self.label_col,
                text_col=self.text_col,
                additional_cols=self.config.get("x_cols", []),
            )
            for fpath in csvs
        ]

    def _resample_positive_rows(self, df: pd.DataFrame, positive_label: Any = "1") -> pd.DataFrame:
        """Upsample the minority positive class when ``positive_resample_rate`` > 1."""
        rate = self.config.get("positive_resample_rate", 1)
        if isinstance(self.label_col, str) and isinstance(rate, int) and rate > 1:
            label_dict = df[self.label_col].value_counts().to_dict()
            if positive_label is None:
                positive_label = min(label_dict, key=label_dict.get)
            logger.info("resampling_positives", rate=rate, label=positive_label)
            return resample_positives(df, rate, self.label_col, positive_label)
        return df

    def split_dataframes(
        self,
        all_dataframes: list[pd.DataFrame] | None = None,
        train_dataframes: list[pd.DataFrame] | None = None,
        val_dataframes: list[pd.DataFrame] | None = None,
        test_dataframes: list[pd.DataFrame] | None = None,
        stage: str | None = None,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
        """Split or assign train/val/test frames; applies resampling and optional ``label_map`` for fit."""
        if stage == "fit" or stage is None:
            if all_dataframes is None:
                train_ds, _, _ = split_dataframes(train_dataframes, 1.0, 0.0, shuffle=True)
                _, val_ds, _ = split_dataframes(val_dataframes, 0.0, 1.0, shuffle=True)
                _, _, test_ds = split_dataframes(test_dataframes, 0.0, 0.0, 1.0, shuffle=True)
            else:
                train_ds, val_ds, test_ds = split_dataframes(
                    all_dataframes,
                    self.config.get("train_ratio", 0.7),
                    self.config.get("validation_ratio", 0.15),
                    shuffle=True,
                )

            train_ds = self._resample_positive_rows(train_ds)

            label_map = self.config.get("label_map")
            if isinstance(label_map, dict):
                train_ds = map_labels(train_ds, self.label_col, label_map)
                val_ds = map_labels(val_ds, self.label_col, label_map)
                test_ds = map_labels(test_ds, self.label_col, label_map)

            logger.info(
                "dataset_split",
                train=train_ds.shape,
                val=val_ds.shape,
                test=test_ds.shape,
            )
            return train_ds, val_ds, test_ds
        else:
            test_ds = pd.concat(all_dataframes if all_dataframes is not None else test_dataframes)
            test_ds[self.sample_id_col] = range(len(test_ds))
            return None, None, test_ds

    def _set_class_sizes(self) -> None:
        """Record per-class counts on the training split for loss weighting."""
        self.train_class_sizes = self._train_dataset[self.label_col].value_counts().to_dict()
        logger.info("class_sizes", sizes=self.train_class_sizes)
