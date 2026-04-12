"""Targeted filters for noisy upstream warnings during sheepy training runs.

Lightning / torchmetrics / W&B sometimes emit warnings that are expected in normal
use (e.g. single-class mini-batches, internal deprecations). Filters are scoped by
message or module so we do not blanket-suppress all warnings.
"""
from __future__ import annotations

import warnings

_applied = False


def apply_sheepy_training_warning_filters() -> None:
    """Register filters once per process (safe to call multiple times)."""
    global _applied
    if _applied:
        return
    _applied = True

    # Lightning 2.x: jaxtyping / pytree LeafSpec deprecation inside the trainer
    warnings.filterwarnings(
        "ignore",
        message=r".*`isinstance\(treespec, LeafSpec\)` is deprecated.*",
        category=DeprecationWarning,
        module=r"lightning.pytorch.utilities._pytree",
    )

    # torchmetrics ROC/AUROC: some train/val mini-batches are single-class (esp. with RandomSampler)
    warnings.filterwarnings(
        "ignore",
        message=r"No (positive|negative) samples in targets, .* should be meaningless\.",
        category=UserWarning,
    )

    try:
        from lightning.pytorch.utilities.rank_zero import PossibleUserWarning
    except Exception:  # pragma: no cover
        PossibleUserWarning = None  # type: ignore[misc, assignment]

    if PossibleUserWarning is not None:
        # Lightning: "Consider increasing num_workers" when DataLoader uses few workers.
        warnings.filterwarnings(
            "ignore",
            message=r"The '.*' does not have many workers which may be a bottleneck\..*",
            category=PossibleUserWarning,
            module=r"lightning.pytorch.trainer.connectors.data_connector",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Consider setting `persistent_workers=True`.*",
            category=PossibleUserWarning,
            module=r"lightning.pytorch.trainer.connectors.data_connector",
        )
