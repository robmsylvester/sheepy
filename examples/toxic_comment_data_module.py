"""Toxic Comments multilabel example data module — updated for sheepy 1.0."""
from __future__ import annotations

import os
from typing import Any

import pandas as pd

from sheepy.common.df_ops import split_dataframes
from sheepy.common.logger import get_logger
from sheepy.common.s3_ops import download_resource
from sheepy.data_modules.multi_label_csv_data_module import MultiLabelCSVDataModule

logger = get_logger(__name__)


class ToxicCommentDataModule(MultiLabelCSVDataModule):
    """Loads the Jigsaw Toxic Comments dataset for multilabel classification."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

    def prepare_data(self, stage: str | None = None) -> None:
        if self.evaluate:
            return
        data_dir = self.config["data_dir"]
        os.makedirs(data_dir, exist_ok=True)

        for fname, key in [
            ("train.csv", "datasets/toxic_comments/train.csv"),
            ("test_filtered.csv", "datasets/toxic_comments/test_filtered.csv"),
        ]:
            dst = os.path.join(data_dir, fname)
            if not os.path.exists(dst):
                logger.info("downloading", file=fname)
                download_resource(key, dst)

    def setup(self, stage: str | None = None) -> None:
        data_dir = self.config["data_dir"]
        if stage == "fit" or stage is None:
            train_dfs = self._read_csv_directory(os.path.join(data_dir, "train.csv"))
            self._train_dataset, self._val_dataset, _ = split_dataframes(
                train_dfs,
                train_ratio=self.config.get("train_ratio", 0.7),
                validation_ratio=self.config.get("validation_ratio", 0.15),
                shuffle=True,
            )

            test_dfs = self._read_csv_directory(os.path.join(data_dir, "test_filtered.csv"))
            _, _, self._test_dataset = split_dataframes(test_dfs, 0.0, 0.0, 1.0, shuffle=True)
            self._train_dataset = self._resample_positive_rows(self._train_dataset)
            self._test_dataset[self.sample_id_col] = range(len(self._test_dataset))

            logger.info(
                "dataset_ready",
                train=self._train_dataset.shape,
                val=self._val_dataset.shape,
                test=self._test_dataset.shape,
            )
        else:
            self._test_dataset = pd.concat(
                self._read_csv_directory(os.path.join(data_dir, "test_filtered.csv"))
            )
            self._test_dataset[self.sample_id_col] = range(len(self._test_dataset))
