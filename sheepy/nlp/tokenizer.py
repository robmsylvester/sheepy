"""Tokenizer wrapper around HuggingFace AutoTokenizer.

Removes the legacy torchnlp TextEncoder dependency.
"""
from __future__ import annotations

from typing import Tuple

import torch
from transformers import AutoTokenizer


class Tokenizer:
    """Wraps ``AutoTokenizer`` with convenience encode / batch_encode methods
    that return ``(input_ids, lengths)`` tensors.
    """

    def __init__(self, pretrained_model: str, max_length: int = 512) -> None:
        """Load ``AutoTokenizer`` and cap encoded sequence length."""
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.max_length = max_length

    # -- Token-index properties ----------------------------------------
    @property
    def unk_index(self) -> int | None:
        """HF unk token id, if defined."""
        return self.tokenizer.unk_token_id

    @property
    def bos_index(self) -> int | None:
        """CLS / sequence-start token id for BERT-style models."""
        return self.tokenizer.cls_token_id

    @property
    def eos_index(self) -> int | None:
        """SEP / segment-boundary token id when present."""
        return self.tokenizer.sep_token_id

    @property
    def pad_index(self) -> int | None:
        """Padding token id for batching."""
        return self.tokenizer.pad_token_id

    @property
    def vocab_size(self) -> int:
        """Tokenizer vocabulary size reported by HuggingFace."""
        return self.tokenizer.vocab_size

    # -- Encoding ------------------------------------------------------
    def encode(self, sequence: str) -> torch.Tensor:
        """Tokenize a single string to 1-D ``input_ids`` (no padding)."""
        return self.tokenizer(sequence, return_tensors="pt")["input_ids"].squeeze(0)

    def batch_encode(self, sentences: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch of sentences.

        Returns:
            Tuple of (input_ids [B, seq_len], lengths [B]).
        """
        out = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            return_length=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation="only_first",
            max_length=self.max_length,
        )
        input_ids = out["input_ids"]
        attention_mask = out["attention_mask"]
        lengths = attention_mask.sum(dim=1)
        return input_ids, lengths
