"""Label encoding for single-label and multilabel classification."""
from __future__ import annotations

import pandas as pd
from torch import Tensor, long, stack, tensor, transpose

from sheepy.common.logger import get_logger

logger = get_logger(__name__)


class LabelEncoder:
    """Maps string labels to integer indices and back.

    Args:
        labels: Ordered list of label strings.
        multilabel: Whether this encoder handles multilabel classification.
        unknown_label: Optional label to use for unseen strings.
    """

    def __init__(
        self,
        labels: list[str],
        multilabel: bool = False,
        unknown_label: str | None = None,
    ) -> None:
        """Build bidirectional maps from ordered *labels* and optional OOV handling."""
        self.multilabel = multilabel
        self.unknown_label = unknown_label
        self.vocab = labels
        self.labels_to_index = {val: idx for idx, val in enumerate(self.vocab)}
        self.index_to_labels = {idx: val for idx, val in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def encode(self, label_str: str) -> Tensor:
        """Encode a single label string to an integer tensor."""
        if label_str not in self.labels_to_index:
            if self.unknown_label is None:
                raise ValueError(f"Unknown label '{label_str}' and no unknown_label is configured.")
            return tensor(self.labels_to_index[self.unknown_label], dtype=long)
        return tensor(self.labels_to_index[label_str], dtype=long)

    def batch_encode(self, label_strings: list[str]) -> Tensor:
        """Encode a batch of label strings to a 1-D integer tensor."""
        return stack([self.encode(s) for s in label_strings], dim=0)

    def batch_encode_multilabel(self, sample: dict) -> Tensor:
        """Encode multilabel columns from a collated sample dict to a float tensor.

        Returns:
            Float tensor of shape ``(batch_size, num_labels)``.
        """
        outputs = []
        for label in self.vocab:
            outputs.append([int(i) for i in sample[label]])
        return transpose(tensor(outputs), 0, 1).float()

    def decode(self, label_tensor: Tensor) -> str:
        """Decode a single integer tensor back to its label string."""
        return self.index_to_labels[label_tensor.squeeze().item()]

    def batch_decode(self, label_tensor: Tensor) -> list[str]:
        """Decode a batch of integer tensors back to label strings."""
        return [self.decode(t.squeeze(0)) for t in label_tensor.split(1, dim=0)]

    def batch_decode_multilabel(self, label_tensor: Tensor) -> dict[str, Tensor]:
        """Decode a multilabel tensor into a dict of per-label tensors."""
        return {label: label_tensor[:, idx] for idx, label in enumerate(self.vocab)}

    @classmethod
    def initializeFromDataframe(
        cls,
        df: pd.DataFrame,
        label_col: str,
        reserved_labels: list[str] | None = None,
        unknown_label: str | None = None,
    ) -> "LabelEncoder":
        """Build a label encoder from unique values in a DataFrame column."""
        labels_list = list(reserved_labels or [])
        for found in df[label_col].unique().tolist():
            if found not in labels_list:
                labels_list.append(found)
        if unknown_label is not None and unknown_label not in labels_list:
            labels_list.append(unknown_label)
        logger.info("label_encoder_created", labels=labels_list)
        return cls(labels_list, multilabel=False, unknown_label=unknown_label)

    @classmethod
    def initializeFromMultilabelDataframe(
        cls,
        df: pd.DataFrame,
        label_list: list[str],
        reserved_labels: list[str] | None = None,
        unknown_label: str | None = None,
    ) -> "LabelEncoder":
        """Build a multilabel encoder from a list of binary label columns."""
        labels_list = list(reserved_labels or [])
        for label in label_list:
            unique_vals = df[label].unique()
            if len(unique_vals) > 2:
                raise ValueError(f"Label '{label}' must be binary, found values: {unique_vals}")
            if label not in labels_list:
                labels_list.append(label)
        if unknown_label is not None and unknown_label not in labels_list:
            labels_list.append(unknown_label)
        return cls(labels_list, multilabel=True, unknown_label=unknown_label)
