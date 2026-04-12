"""Collation utilities for batching samples into model inputs.

Replaces the legacy torchnlp ``collate_tensors`` with a pure-Python implementation.
"""
from __future__ import annotations

import random
from collections import namedtuple
from typing import Any

import torch
from torch import Tensor

from sheepy.nlp.label_encoder import LabelEncoder
from sheepy.nlp.tokenizer import Tokenizer

CollatedSample = namedtuple("CollatedSample", ["inputs", "targets", "ids"])


def collate_tensors(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Transpose a list of dicts into a dict of lists (replaces ``torchnlp.utils.collate_tensors``)."""
    keys = batch[0].keys()
    return {k: [sample[k] for sample in batch] for k in keys}


def single_text_collate_function(
    sample: list[dict],
    text_key: str,
    label_keys: str | list[str],
    id_key: str,
    tokenizer: Tokenizer,
    label_encoder: LabelEncoder | None = None,
    evaluate: bool = False,
) -> CollatedSample:
    """Collate a batch of raw dict samples into tensors for the model.

    Returns:
        ``CollatedSample(inputs, targets, ids)`` where *inputs* contains
        ``tokens`` and ``lengths``, *targets* contains ``labels``, and *ids*
        contains ``sample_id_keys`` (only in evaluate mode).
    """
    sample = collate_tensors(sample)

    try:
        tokens, lengths = tokenizer.batch_encode(sample[text_key])
    except KeyError:
        raise KeyError(
            f"Text key '{text_key}' does not exist in sample. Keys: {list(sample.keys())}"
        )

    inputs = {"tokens": tokens, "lengths": lengths}
    targets: dict[str, Tensor] = {}
    ids: dict[str, Tensor] = {}

    if evaluate:
        ids = {"sample_id_keys": torch.tensor(sample[id_key], dtype=torch.long)}
    else:
        try:
            if isinstance(label_keys, list):
                targets = {"labels": label_encoder.batch_encode_multilabel(sample)}
            elif isinstance(label_keys, str):
                targets = {"labels": label_encoder.batch_encode(sample[label_keys])}
        except KeyError:
            raise KeyError(
                f"Label key '{label_keys}' does not exist in sample. Keys: {list(sample.keys())}"
            )
        except RuntimeError:
            raise RuntimeError("Label encoder found an unknown label.")

    return CollatedSample(inputs, targets, ids)


def _get_sample_list(samples: list[str], sample_size: int | None = None) -> list[str]:
    """Return a random subset (without replacement) from *samples*."""
    if sample_size is None:
        sample_size = len(samples)
    if sample_size > len(samples):
        raise ValueError(
            f"Cannot sample {sample_size} items from {len(samples)} available."
        )
    return random.sample(samples, k=sample_size)


def _concatenate_text_samples(
    sample: dict[str, list[str]],
    keys: list[str],
    default_separator: str = ".",
) -> list[str]:
    """Concatenate multiple text columns from a collated sample into single strings."""
    out_strings = []
    batch_size = len(sample[keys[0]])
    for idx in range(batch_size):
        parts = []
        for key in keys:
            text = sample[key][idx]
            if text:
                if text[-1] not in ("?", "!", "."):
                    text += default_separator
                parts.append(text)
        out_strings.append(" ".join(parts))
    return out_strings


def round_size(size: int) -> int:
    """Round *size* up to the next power of two."""
    p = 2
    while p < size:
        p *= 2
    return p
