"""Base data module for sheepy experiments, built on Lightning 2.x."""
from __future__ import annotations

import os
from typing import Any

import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler

from sheepy.common.collate import CollatedSample, single_text_collate_function
from sheepy.common.logger import get_logger
from sheepy.nlp.label_encoder import LabelEncoder
from sheepy.nlp.tokenizer import Tokenizer

logger = get_logger(__name__)


class BaseDataModule(L.LightningDataModule):
    """Abstract data module that defines the standard interface.

    Args:
        config: Flat dict of experiment configuration.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Store config, output dir, tokenizer reference, and unset dataset handles."""
        super().__init__()
        self.config = config
        self.sample_id_col = "sample_id"
        self.evaluate = config.get("evaluate", False)
        self.label_col = config.get("label", "label")
        self.text_col = config.get("text", "text")

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self._predict_dataset = None

        output_dir = config.get("output_dir", ".")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.tokenizer = Tokenizer(config.get("encoder_model", "bert-base-uncased"))
        self.label_encoder: LabelEncoder | None = None
        self.train_class_sizes: dict[str, int] | None = None

    def prepare_data(self) -> None:
        """Subclasses must implement dataset download or preprocessing."""
        raise NotImplementedError("Subclass must implement prepare_data().")

    def setup(self, stage: str | None = None) -> None:
        """Override to assign ``_train_dataset`` / ``_val_dataset`` / ``_test_dataset`` for *stage*."""
        pass

    def _verify_module(self) -> None:
        """Hook after setup; CSV modules build encoders and class counts here."""
        pass

    def save_data_module(self, out_path: str | None = None) -> None:
        """Override to persist encoder metadata needed to reload checkpoints."""
        pass

    def load_data_module(self, in_path: str | None = None) -> None:
        """Override to restore encoder metadata saved by ``save_data_module``."""
        pass

    def _dataloader_kwargs(self) -> dict[str, Any]:
        """Common DataLoader options: workers, ``persistent_workers``, ``pin_memory``."""
        nw = int(self.config.get("loader_workers", 0))
        kw: dict[str, Any] = {"num_workers": nw}
        if nw > 0:
            kw["persistent_workers"] = True
        kw["pin_memory"] = bool(torch.cuda.is_available())
        return kw

    def prepare_sample(self, sample: list[dict]) -> CollatedSample:
        """Collate a batch of raw dict rows into token tensors and labels."""
        if self.text_col is None:
            raise NotImplementedError("A text column is required for the default collate function.")
        return single_text_collate_function(
            sample,
            self.text_col,
            self.label_col,
            self.sample_id_col,
            self.tokenizer,
            self.label_encoder,
            evaluate=self.evaluate,
        )

    def _write_predictions(self, outputs: list[dict]) -> None:
        """Aggregate ``predict_step`` logits and write softmax probabilities to CSV."""
        all_columns = list(self.label_encoder.vocab)
        all_columns.append(self.sample_id_col)
        frames = []

        for output in outputs:
            prediction_logits = output["logits"].cpu().squeeze()
            sample_ids = output["sample_id_keys"].cpu().squeeze()
            if prediction_logits.dim() == 1:
                prediction_logits = prediction_logits.unsqueeze(0)

            prediction_softmax = torch.nn.Softmax(dim=1)(prediction_logits)
            batch_df = pd.DataFrame(prediction_softmax.numpy()).astype(float)
            batch_df = batch_df.rename(
                columns={i: label for i, label in enumerate(self.label_encoder.vocab)}
            )
            batch_df[self.sample_id_col] = sample_ids.numpy().astype(int)
            frames.append(batch_df)

        result = pd.concat(frames, ignore_index=True)
        cols = [self.sample_id_col] + [c for c in result.columns if c != self.sample_id_col]
        result = result[cols]

        out_path = self.config.get("output_prediction_path", "predictions.csv")
        logger.info("writing_predictions", count=len(result), path=out_path)
        result.to_csv(out_path, index=False)

    def _verify_dataset_record_type(self, dataset) -> list[dict]:
        """Normalize *dataset* to a non-empty list of dict rows for DataLoader."""
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.to_dict("records")
        assert isinstance(dataset, list) and len(dataset) > 0 and isinstance(dataset[0], dict)
        return dataset

    def _load_text_from_text_file(self, filepath: str) -> list[dict]:
        """Build one sample dict per non-empty line for batch evaluation."""
        prepared = []
        with open(filepath, "r") as f:
            for idx, line in enumerate(f):
                prepared.append({
                    self.text_col: line.rstrip(),
                    self.label_col: None,
                    self.sample_id_col: idx,
                })
        return prepared

    def _load_text_from_raw_input(self) -> list[dict]:
        """Prompt stdin once and return a single-row sample for live prediction."""
        text = input("Enter sample text (Ctrl+C to exit):\n")
        return [{self.text_col: text, self.label_col: None, self.sample_id_col: 0}]

    def train_dataloader(self) -> DataLoader:
        """Random-sampled training loader with ``prepare_sample`` collate."""
        ds = self._verify_dataset_record_type(self._train_dataset)
        return DataLoader(
            dataset=ds,
            sampler=RandomSampler(ds),
            batch_size=self.config.get("batch_size", 16),
            collate_fn=self.prepare_sample,
            **self._dataloader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        """Sequential validation loader."""
        ds = self._verify_dataset_record_type(self._val_dataset)
        return DataLoader(
            dataset=ds,
            batch_size=self.config.get("batch_size", 16),
            collate_fn=self.prepare_sample,
            **self._dataloader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        """Sequential test loader."""
        ds = self._verify_dataset_record_type(self._test_dataset)
        return DataLoader(
            dataset=ds,
            batch_size=self.config.get("batch_size", 16),
            collate_fn=self.prepare_sample,
            **self._dataloader_kwargs(),
        )

    def predict_batch_dataloader(self) -> DataLoader:
        """Loader over lines from ``evaluate_batch_file`` for offline inference."""
        self._predict_dataset = self._load_text_from_text_file(self.config["evaluate_batch_file"])
        return DataLoader(
            dataset=self._predict_dataset,
            batch_size=self.config.get("batch_size", 16),
            collate_fn=self.prepare_sample,
            **self._dataloader_kwargs(),
        )

    def predict_live_dataloader(self) -> DataLoader:
        """Single-sample loader driven by interactive stdin for live inference."""
        self._predict_dataset = self._load_text_from_raw_input()
        return DataLoader(
            dataset=self._predict_dataset,
            batch_size=1,
            collate_fn=self.prepare_sample,
            num_workers=0,
        )

    def _build_label_encoder(self) -> None:
        """Fit ``LabelEncoder`` on the training dataframe's label column."""
        self.label_encoder = LabelEncoder.initializeFromDataframe(
            self._train_dataset, self.label_col
        )
        logger.info("label_encoder_built", vocab=self.label_encoder.vocab)

    def _set_class_sizes(self) -> None:
        """Subclasses must implement per-class counts (or per-label counts) for weighting."""
        raise NotImplementedError("Subclass must implement _set_class_sizes().")
