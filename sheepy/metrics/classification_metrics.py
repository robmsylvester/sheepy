"""Classification metrics built on torchmetrics v1+ API.

Supports binary, multiclass, and multilabel tasks via the ``task`` parameter.
"""
from __future__ import annotations

from typing import Any

import torch
from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    F1Score,
    MetricCollection,
    Precision,
    Recall,
)

from sheepy.common.logger import get_logger

logger = get_logger(__name__)

TORCHMETRICS_MAP: dict[str, type] = {
    "accuracy": Accuracy,
    "auroc": AUROC,
    "f1": F1Score,
    "precision": Precision,
    "recall": Recall,
    "average_precision": AveragePrecision,
}

# Maps the short user-facing config name → the stable key logged to W&B.
# These keys are what users write in `validation.metric` in their config.
METRIC_KEY_MAP: dict[str, str] = {
    "accuracy": "Accuracy",
    "auroc": "AUROC",
    "f1": "F1Score",
    "precision": "Precision",
    "recall": "Recall",
    "average_precision": "AveragePrecision",
}


class ClassificationMetrics:
    """Thin wrapper around ``torchmetrics.MetricCollection`` that auto-creates
    train / val / test copies and dispatches ``compute_metrics`` by stage.

    Args:
        num_labels: Number of classes / labels.
        config: Metrics configuration dict (from ``MetricsConfig.model_dump()``).
        task: One of ``"binary"``, ``"multiclass"``, or ``"multilabel"``.
    """

    def __init__(
        self,
        num_labels: int,
        config: dict[str, Any],
        task: str | None = None,
    ) -> None:
        """Infer ``task`` when omitted and build train/val/test metric collections."""
        self.num_labels = num_labels
        self.config = config

        if task is None:
            if config.get("multilabel", False):
                task = "multilabel"
            elif num_labels > 2:
                task = "multiclass"
            else:
                task = "binary"
        self.task = task

        self._create_metrics()

    # ------------------------------------------------------------------
    def _build_metric(self, metric_def: dict[str, Any]) -> Any:
        """Instantiate one torchmetrics metric from a tracked-metric definition, or None if unknown."""
        name = metric_def["name"]
        cls = TORCHMETRICS_MAP.get(name)
        if cls is None:
            logger.warning("metric_not_found", metric=name)
            return None

        kwargs: dict[str, Any] = {"task": self.task}
        if self.task == "multiclass":
            kwargs["num_classes"] = self.num_labels
        elif self.task == "multilabel":
            kwargs["num_labels"] = self.num_labels
        elif self.task == "binary":
            pass  # no extra dim args needed

        avg = metric_def.get("average", "macro")
        if name not in ("accuracy",):
            kwargs["average"] = avg

        # Fewer batch-level UserWarnings (e.g. single-class mini-batches) while keeping numerics.
        if name in ("auroc", "f1", "precision", "recall", "average_precision"):
            kwargs["validate_args"] = False

        return cls(**kwargs)

    def _create_metrics(self) -> None:
        """Populate ``metric_collection`` and prefixed train/val/test clones with stable W&B keys."""
        # Build a dict so MetricCollection uses our stable keys (e.g. "F1Score")
        # instead of the torchmetrics class names (e.g. "BinaryF1Score").
        # This means W&B always receives keys like val/F1Score regardless of task,
        # which matches what users write in validation.metric in their config.
        metrics: dict[str, Any] = {}
        for mdef in self.config.get("tracked_metrics", []):
            m = self._build_metric(mdef)
            if m is not None:
                key = METRIC_KEY_MAP.get(mdef["name"], mdef["name"].capitalize())
                metrics[key] = m

        if not metrics:
            metrics["Accuracy"] = Accuracy(
                task=self.task,
                num_classes=self.num_labels if self.task == "multiclass" else None,
                num_labels=self.num_labels if self.task == "multilabel" else None,
            )

        self.metric_collection = MetricCollection(metrics)
        self.train_metrics = self.metric_collection.clone(prefix="train/")
        self.validation_metrics = self.metric_collection.clone(prefix="val/")
        self.test_metrics = self.metric_collection.clone(prefix="test/")

    # ------------------------------------------------------------------
    def compute_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        stage: str,
    ) -> dict[str, torch.Tensor]:
        """Apply task-appropriate probabilities and run the metric bundle for *stage*."""
        preds = logits.detach().cpu()
        targets = labels.detach().cpu()

        if self.task == "multilabel":
            # sigmoid → [N, L] float probabilities
            preds = torch.sigmoid(preds)
        elif self.task == "multiclass":
            # softmax → [N, C] class probabilities
            preds = torch.softmax(preds, dim=-1)
        elif self.task == "binary":
            # torchmetrics binary metrics expect [N] — probability of the positive class
            preds = torch.softmax(preds, dim=-1)[:, 1]

        collection = {
            "train": self.train_metrics,
            "train_epoch": self.train_metrics,
            "val": self.validation_metrics,
            "val_epoch": self.validation_metrics,
            "test": self.test_metrics,
            "test_epoch": self.test_metrics,
        }.get(stage)

        if collection is None:
            raise ValueError(f"Unknown stage '{stage}'. Expected train/val/test.")

        return collection(preds, targets)

    def reset(self, stage: str | None = None) -> None:
        """Reset internal metric state for *stage*, or all stages when *stage* is None."""
        if stage is None or stage == "train":
            self.train_metrics.reset()
        if stage is None or stage == "val":
            self.validation_metrics.reset()
        if stage is None or stage == "test":
            self.test_metrics.reset()
