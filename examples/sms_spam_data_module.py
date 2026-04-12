"""SMS Spam example data module — updated for sheepy 1.0 config dict API."""
from __future__ import annotations

import os
from typing import Any

import pandas as pd

from sheepy.common.collate import CollatedSample, single_text_collate_function
from sheepy.common.logger import get_logger
from sheepy.common.s3_ops import download_resource
from sheepy.data_modules.base_csv_data_module import BaseCSVDataModule

logger = get_logger(__name__)


class SmsSpamDataModule(BaseCSVDataModule):
    """Loads the SMS Spam dataset for binary text classification."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.all_dataframes = None

    def _read_dataset(self) -> pd.DataFrame:
        rows = []
        path = os.path.join(self.config["data_dir"], "SMSSpamCollection")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    rows.append({self.label_col: parts[0], self.text_col: parts[1]})
        return pd.DataFrame(rows)

    def prepare_data(self, stage: str | None = None) -> None:
        if self.evaluate:
            return
        data_dir = self.config["data_dir"]
        os.makedirs(data_dir, exist_ok=True)
        sms_path = os.path.join(data_dir, "SMSSpamCollection")
        if not os.path.exists(sms_path):
            logger.info("downloading_sms_spam")
            download_resource("datasets/spam/SMSSpamCollection", sms_path)

    def setup(self, stage: str | None = None) -> None:
        if self.evaluate:
            data_dir = self.config["data_dir"]
            self.all_dataframes = []
            for f in os.listdir(data_dir):
                if f.endswith(".txt"):
                    df = pd.read_csv(os.path.join(data_dir, f), header=None, names=[self.text_col])
                    self.all_dataframes.append(df)
        else:
            self.all_dataframes = [self._read_dataset()]

        self._train_dataset, self._val_dataset, self._test_dataset = self.split_dataframes(
            all_dataframes=self.all_dataframes, stage=stage
        )

    def prepare_sample(self, sample: list[dict]) -> CollatedSample:
        return single_text_collate_function(
            sample, self.text_col, self.label_col, self.sample_id_col,
            self.tokenizer, self.label_encoder, evaluate=self.evaluate,
        )
