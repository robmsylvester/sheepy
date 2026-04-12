"""W&B Artifact utilities for dataset versioning."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import wandb

from sheepy.common.logger import get_logger

logger = get_logger(__name__)


def log_dataset_artifact(
    run: wandb.sdk.wandb_run.Run | Any,
    name: str,
    data_paths: dict[str, str | Path],
    metadata: dict[str, Any] | None = None,
) -> wandb.Artifact:
    """Log train/val/test splits as a versioned W&B Artifact.

    Args:
        run: Active ``wandb.Run`` (usually ``self.logger.experiment`` in Lightning).
        name: Artifact name (e.g. ``"sms_spam_dataset"``).
        data_paths: Dict mapping split name to file path,
            e.g. ``{"train": "data/train.csv", "val": "data/val.csv"}``.
        metadata: Optional metadata dict attached to the artifact.

    Returns:
        The logged ``wandb.Artifact``.
    """
    artifact = wandb.Artifact(name=name, type="dataset", metadata=metadata or {})
    for split_name, path in data_paths.items():
        path = Path(path)
        if path.exists():
            artifact.add_file(str(path), name=f"{split_name}{path.suffix}")
        else:
            logger.warning("artifact_file_missing", split=split_name, path=str(path))
    run.log_artifact(artifact)
    logger.info("dataset_artifact_logged", name=name, splits=list(data_paths.keys()))
    return artifact


def log_model_artifact(
    run: wandb.sdk.wandb_run.Run | Any,
    name: str,
    model_dir: str | Path,
    metadata: dict[str, Any] | None = None,
) -> wandb.Artifact:
    """Log a trained model directory as a W&B Artifact.

    Args:
        run: Active ``wandb.Run``.
        name: Artifact name.
        model_dir: Directory containing the model files.
        metadata: Optional metadata.

    Returns:
        The logged ``wandb.Artifact``.
    """
    artifact = wandb.Artifact(name=name, type="model", metadata=metadata or {})
    model_dir = Path(model_dir)
    if model_dir.is_dir():
        artifact.add_dir(str(model_dir))
    else:
        artifact.add_file(str(model_dir))
    run.log_artifact(artifact)
    logger.info("model_artifact_logged", name=name)
    return artifact
