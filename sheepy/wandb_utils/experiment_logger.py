"""High-level experiment logger that wraps W&B with rich visualizations."""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import wandb

from sheepy.common.logger import get_logger
from sheepy.wandb_utils.tables import (
    build_dataset_summary_table,
    build_per_class_metrics_dict,
    build_prediction_table,
)

logger = get_logger(__name__)


def _labels_to_class_indices(labels: Sequence[Any], class_names: list[str]) -> list[int]:
    """Convert labels to ints for ``wandb.plot.confusion_matrix``.

    W&B builds an internal ``class_mapping`` from ``range(len(class_names))`` (integers),
    not from the name strings, so passing string labels causes ``KeyError``. This maps
    each label to ``class_names`` index; integer labels are validated and passed through.
    """
    name_to_idx = {str(name): i for i, name in enumerate(class_names)}
    n = len(class_names)
    out: list[int] = []
    for v in labels:
        if isinstance(v, (bool, np.bool_)):
            v = int(v)
        if isinstance(v, (int, np.integer)):
            iv = int(v)
            if not (0 <= iv < n):
                raise ValueError(
                    f"Class index {iv} out of range for {n} classes {class_names!r}."
                )
            out.append(iv)
            continue
        key = str(v)
        if key not in name_to_idx:
            raise ValueError(
                f"Label {key!r} is not in class_names={class_names!r}. "
                "Pass strings matching those names or int indices in [0, n_classes)."
            )
        out.append(name_to_idx[key])
    return out


class ExperimentLogger:
    """Provides structured W&B logging beyond basic scalar metrics.

    Designed to be used alongside the Lightning ``WandbLogger``.

    Args:
        run: The active ``wandb.Run`` (typically ``trainer.logger.experiment``).
        class_names: List of class/label names.
    """

    def __init__(self, run: Any, class_names: list[str]) -> None:
        """Bind an active W&B run and ordered class names for tables and confusion matrices."""
        self.run = run
        self.class_names = class_names

    def log_experiment_metadata(self, config: dict[str, Any]) -> None:
        """Log git commit, config hash, and other metadata to the run config."""
        meta: dict[str, Any] = {}
        try:
            meta["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            meta["git_commit"] = "unknown"

        config_str = json.dumps(config, sort_keys=True, default=str)
        meta["config_hash"] = hashlib.sha256(config_str.encode()).hexdigest()[:12]

        if self.run:
            self.run.config.update(meta)

    def log_dataset_summary(self, label_counts: dict[str, int]) -> None:
        """Log a dataset summary table to W&B."""
        if not self.run:
            return
        # Re-logging the same table on resumed runs creates a duplicate-artifact warning.
        if self.run.config.get("_sheepy_dataset_summary_logged"):
            logger.info("dataset_summary_skipped_already_logged")
            return
        table = build_dataset_summary_table(label_counts)
        self.run.log({"dataset/summary": table})
        try:
            self.run.config.update({"_sheepy_dataset_summary_logged": True}, allow_val_change=True)
        except TypeError:
            self.run.config["_sheepy_dataset_summary_logged"] = True

    def log_predictions(
        self,
        texts: list[str],
        true_labels: list[str],
        predicted_labels: list[str],
        confidences: list[float],
        stage: str = "val",
    ) -> None:
        """Log a full prediction table to W&B."""
        table = build_prediction_table(texts, true_labels, predicted_labels, confidences)
        if self.run:
            self.run.log({f"{stage}/predictions": table})

    def log_confusion_matrix(
        self,
        y_true: list | np.ndarray,
        y_pred: list | np.ndarray,
        stage: str = "val",
    ) -> None:
        """Log a confusion matrix to W&B.

        ``y_true`` / ``y_pred`` may be integer class indices or strings matching
        ``self.class_names`` (e.g. decoded label names from a ``LabelEncoder``).
        """
        if isinstance(y_true, np.ndarray):
            y_true = y_true.tolist()
        if isinstance(y_pred, np.ndarray):
            y_pred = y_pred.tolist()

        y_true_i = _labels_to_class_indices(y_true, self.class_names)
        y_pred_i = _labels_to_class_indices(y_pred, self.class_names)

        cm = wandb.plot.confusion_matrix(
            y_true=y_true_i,
            preds=y_pred_i,
            class_names=self.class_names,
        )
        if self.run:
            self.run.log({f"{stage}/confusion_matrix": cm})

    def log_pr_curve(
        self,
        y_true: np.ndarray,
        y_probas: np.ndarray,
        stage: str = "val",
    ) -> None:
        """Log precision-recall curves to W&B."""
        if self.run:
            self.run.log({
                f"{stage}/pr_curve": wandb.plot.pr_curve(
                    y_true, y_probas, labels=self.class_names
                )
            })

    def log_roc_curve(
        self,
        y_true: np.ndarray,
        y_probas: np.ndarray,
        stage: str = "val",
    ) -> None:
        """Log ROC curves to W&B."""
        if self.run:
            self.run.log({
                f"{stage}/roc_curve": wandb.plot.roc_curve(
                    y_true, y_probas, labels=self.class_names
                )
            })

    def log_per_class_metrics(
        self,
        metric_name: str,
        values: list[float] | np.ndarray | torch.Tensor,
        stage: str = "val",
    ) -> None:
        """Log per-class metric values as individual scalars."""
        metrics = build_per_class_metrics_dict(self.class_names, metric_name, values, prefix=stage)
        if self.run:
            self.run.log(metrics)

    def update_summary(self, key: str, value: Any) -> None:
        """Update ``wandb.summary`` with a best-metric value."""
        if self.run:
            self.run.summary[key] = value
