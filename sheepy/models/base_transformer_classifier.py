"""Transformer-based text classifier using HuggingFace AutoModelForSequenceClassification.

Handles binary, multiclass, and multilabel tasks through a single class.
"""
from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch import optim
from transformers import AutoConfig, AutoModelForSequenceClassification

from sheepy.models.base_classifier import BaseClassifier


class TransformerClassifier(BaseClassifier):
    """Sequence classifier backed by any HuggingFace model that supports
    ``AutoModelForSequenceClassification``.

    Args:
        config: Flat experiment config dict.
        data: LightningDataModule with label_encoder, tokenizer, etc.
        task: ``"binary"``, ``"multiclass"``, or ``"multilabel"``.
    """

    def __init__(
        self,
        config: dict[str, Any],
        data: L.LightningDataModule,
        task: str | None = None,
    ) -> None:
        """Build HF sequence-classification head and task loss after base setup."""
        super().__init__(config, data, task=task)
        self._build_model()
        self._build_loss()

    def _build_model(self) -> None:
        """Load ``AutoModelForSequenceClassification`` with label ids and problem type."""
        vocab = self.data.label_encoder.vocab
        hf_config = AutoConfig.from_pretrained(
            self.config["encoder_model"],
            num_labels=len(vocab),
            id2label={i: label for i, label in enumerate(vocab)},
            label2id={label: i for i, label in enumerate(vocab)},
            problem_type=(
                "multi_label_classification" if self.task == "multilabel" else "single_label_classification"
            ),
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config["encoder_model"],
            config=hf_config,
        )

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        """Trim to max length in batch, apply attention mask, return classifier logits."""
        tokens = tokens[:, : lengths.max()]
        attention_mask = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0) < lengths.unsqueeze(1)
        outputs = self.model(input_ids=tokens, attention_mask=attention_mask)
        return {"logits": outputs.logits}

    def configure_optimizers(self):
        """AdamW on encoder + head with separate encoder LR entry."""
        parameters = [
            {"params": self.model.parameters(), "lr": self.config.get("encoder_learning_rate", 1e-5)},
        ]
        optimizer = optim.AdamW(parameters, lr=self.config.get("learning_rate", 3e-5))
        return [optimizer], []
