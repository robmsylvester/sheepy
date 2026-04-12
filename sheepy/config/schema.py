"""Pydantic v2 configuration models for sheepy experiments.

Supports loading from JSON, YAML, and TOML with backward compatibility
for the original JSON config format.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ExperimentConfig(BaseModel):
    project_name: str = "default_project"
    description: str = ""
    dataset: str = ""
    version: str | int = "0"
    data_module: str = "base_csv_data_module"
    model: str = "transformer_classifier"
    restore_from_checkpoint: bool = False


class MetricDefinition(BaseModel):
    name: str
    average: str = "macro"
    task: str | None = None


class MetricsConfig(BaseModel):
    multilabel: bool = False
    tracked_metrics: list[MetricDefinition] = Field(default_factory=list)


class HparamsConfig(BaseModel):
    label: str | list[str] = "label"
    text: str | None = "text"
    positive_resample_rate: int | dict[str, int] = 1
    disable_weight_scale: bool = False
    nr_frozen_epochs: int = 0
    loader_workers: int = 4
    encoder_model: str = "bert-base-uncased"
    batch_size: int = 16
    num_epochs: int = 10
    max_epochs: int | None = None
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    dropout_p: float = 0.1
    early_stop_enabled: bool = False
    learning_rate: float = 3e-5
    encoder_learning_rate: float = 1e-5
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    num_labels: int = 2
    unknown_index: int | None = None
    label_map: dict[str, str] | None = None
    n_prev_text_samples_sample_size: int | None = None
    n_next_text_samples_sample_size: int | None = None
    save_epoch_count: int | None = None
    num_epochs_decay: int | None = None
    # Lightning hardware (optional). When unset, sheepy picks CUDA → MPS → CPU.
    accelerator: str | None = Field(
        default=None,
        description="Lightning accelerator, e.g. 'cuda', 'cpu', 'mps'. None = auto from hardware.",
    )
    devices: int | str | None = Field(
        default=None,
        description="Lightning devices: GPU count, or 'auto'. None defaults to 1 CUDA GPU when available.",
    )

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def _set_max_epochs(self) -> "HparamsConfig":
        """Default ``max_epochs`` from ``num_epochs`` when unset."""
        if self.max_epochs is None:
            self.max_epochs = self.num_epochs
        return self


class ValidationConfig(BaseModel):
    metric: str = "val/F1Score"  # must match a key logged via self.log() — see METRIC_KEY_MAP
    metric_goal: str = "max"
    check_val_every_n_epoch: int = 1
    val_check_interval: float = 1.0
    gradient_log_steps: int | None = None
    param_log_steps: int | None = None

    model_config = {"extra": "allow"}


class SweepConfig(BaseModel):
    name: str = "sweep_v1"
    method: str = "bayes"
    metric: dict[str, str] = Field(default_factory=lambda: {"name": "val/F1Score", "goal": "maximize"})
    parameters: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class SheepyConfig(BaseModel):
    """Top-level configuration container for a sheepy experiment."""

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    hparams: HparamsConfig = Field(default_factory=HparamsConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    sweep: SweepConfig = Field(default_factory=SweepConfig)

    model_config = {"extra": "allow"}

    @classmethod
    def from_file(cls, path: str | Path) -> "SheepyConfig":
        """Load config from JSON, YAML, or TOML file."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")

        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        elif path.suffix == ".toml":
            import tomllib
            data = tomllib.loads(text)
        else:
            data = json.loads(text)

        return cls.model_validate(data)

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten config sections into a single dict (backward compat with argparse Namespace)."""
        flat: dict[str, Any] = {}
        flat.update(self.experiment.model_dump())
        flat.update(self.hparams.model_dump())
        flat.update(self.validation.model_dump())
        flat["metrics"] = self.metrics.model_dump()
        flat["sweep"] = self.sweep.model_dump()
        return flat

    @classmethod
    def from_legacy_json(cls, path: str | Path) -> "SheepyConfig":
        """Load from the original sheepy JSON format."""
        return cls.from_file(path)
