"""W&B Table builders for prediction analysis and dataset summaries."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import wandb

from sheepy.common.logger import get_logger

logger = get_logger(__name__)


def build_prediction_table(
    texts: list[str],
    true_labels: list[str],
    predicted_labels: list[str],
    confidences: list[float],
    sample_ids: list[Any] | None = None,
) -> wandb.Table:
    """Create a W&B Table of predictions for interactive analysis.

    Columns: sample_id, text, true_label, predicted_label, confidence, correct.
    """
    columns = ["sample_id", "text", "true_label", "predicted_label", "confidence", "correct"]
    data = []
    for i in range(len(texts)):
        sid = sample_ids[i] if sample_ids else i
        correct = true_labels[i] == predicted_labels[i]
        data.append([sid, texts[i], true_labels[i], predicted_labels[i], confidences[i], correct])
    return wandb.Table(columns=columns, data=data)


def build_dataset_summary_table(
    label_counts: dict[str, int],
    text_lengths: list[int] | None = None,
    sample_texts: list[str] | None = None,
) -> wandb.Table:
    """Create a W&B Table summarizing dataset characteristics.

    Includes label distribution, and optionally text length stats and example texts.
    """
    columns = ["label", "count", "fraction"]
    total = sum(label_counts.values())
    data = []
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        data.append([label, count, count / total if total > 0 else 0])

    table = wandb.Table(columns=columns, data=data)
    return table


def build_per_class_metrics_dict(
    class_names: list[str],
    metric_name: str,
    values: list[float] | np.ndarray | torch.Tensor,
    prefix: str = "val",
) -> dict[str, float]:
    """Build a flat dict of per-class metric values for ``self.log_dict()``.

    Returns keys like ``val/F1/class_spam``, ``val/F1/class_ham``.
    """
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()
    elif isinstance(values, list):
        values = np.array(values)

    return {
        f"{prefix}/{metric_name}/{cls}": float(values[i])
        for i, cls in enumerate(class_names)
    }
