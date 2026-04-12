"""SemEval Sentiment multiclass example data module — updated for sheepy 1.0."""
from __future__ import annotations

import os
from typing import Any

import pandas as pd

from sheepy.common.collate import CollatedSample, single_text_collate_function
from sheepy.common.df_ops import read_csv_text_classifier
from sheepy.common.logger import get_logger
from sheepy.common.s3_ops import download_resource
from sheepy.data_modules.base_csv_data_module import BaseCSVDataModule

logger = get_logger(__name__)


class SemEvalSentimentDataModule(BaseCSVDataModule):
    """Loads the SemEval-2017 sentiment dataset for multiclass classification."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.all_dataframes = None

    def prepare_data(self, stage: str | None = None) -> None:
        if self.evaluate:
            return
        for key_name, config_key, s3_key in [
            ("train", "train_data_dir", "datasets/semeval/semeval-2017-train.csv"),
            ("val", "val_data_dir", "datasets/semeval/semeval-2017-val.csv"),
            ("test", "test_data_dir", "datasets/semeval/semeval-2017-test.csv"),
        ]:
            path = self.config.get(config_key, "")
            if path and (not os.path.isfile(path)):
                logger.info("downloading", split=key_name)
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                download_resource(s3_key, path)

    def setup(self, stage: str | None = None) -> None:
        train_dfs = self._read_csv_directory_tsv(self.config["train_data_dir"])
        val_dfs = self._read_csv_directory_tsv(self.config["val_data_dir"])
        test_dfs = self._read_csv_directory_tsv(self.config["test_data_dir"])

        self._train_dataset, self._val_dataset, self._test_dataset = self.split_dataframes(
            train_dataframes=train_dfs,
            val_dataframes=val_dfs,
            test_dataframes=test_dfs,
            stage=stage,
        )

    def _read_csv_directory_tsv(self, csv_path: str) -> list[pd.DataFrame]:
        if os.path.isdir(csv_path):
            files = [os.path.join(csv_path, f) for f in os.listdir(csv_path) if f.endswith(".csv")]
        elif csv_path.endswith(".csv"):
            files = [csv_path]
        else:
            raise ValueError(f"'{csv_path}' is not a CSV file or directory.")
        return [
            read_csv_text_classifier(
                f, evaluate=self.evaluate, delimiter="\t",
                label_cols=self.label_col, text_col=self.text_col,
            )
            for f in files
        ]

    def prepare_sample(self, sample: list[dict]) -> CollatedSample:
        return single_text_collate_function(
            sample, self.text_col, self.label_col, self.sample_id_col,
            self.tokenizer, self.label_encoder, evaluate=self.evaluate,
        )
