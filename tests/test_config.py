"""Tests for sheepy.config Pydantic models."""
import json
import tempfile
from pathlib import Path

import pytest

from sheepy.config.schema import (
    ExperimentConfig,
    HparamsConfig,
    MetricsConfig,
    SheepyConfig,
    ValidationConfig,
)


class TestSheepyConfig:
    def test_defaults(self):
        cfg = SheepyConfig()
        assert cfg.experiment.project_name == "default_project"
        assert cfg.hparams.batch_size == 16
        assert cfg.hparams.learning_rate == 3e-5

    def test_from_dict(self):
        data = {
            "experiment": {"project_name": "test_project"},
            "hparams": {"batch_size": 32, "encoder_model": "bert-base-uncased"},
            "metrics": {"multilabel": False, "tracked_metrics": []},
            "validation": {"metric": "val/loss"},
            "sweep": {"name": "test_sweep"},
        }
        cfg = SheepyConfig.model_validate(data)
        assert cfg.experiment.project_name == "test_project"
        assert cfg.hparams.batch_size == 32

    def test_to_flat_dict(self):
        cfg = SheepyConfig()
        flat = cfg.to_flat_dict()
        assert "project_name" in flat
        assert "batch_size" in flat
        assert "metrics" in flat
        assert "sweep" in flat

    def test_from_json_file(self, tmp_path):
        data = {
            "experiment": {"project_name": "json_test"},
            "hparams": {"num_labels": 3},
            "metrics": {"multilabel": False, "tracked_metrics": []},
            "validation": {},
            "sweep": {},
        }
        f = tmp_path / "config.json"
        f.write_text(json.dumps(data))
        cfg = SheepyConfig.from_file(f)
        assert cfg.experiment.project_name == "json_test"
        assert cfg.hparams.num_labels == 3

    def test_from_yaml_file(self, tmp_path):
        yaml_content = """
experiment:
  project_name: yaml_test
hparams:
  batch_size: 64
metrics:
  multilabel: false
  tracked_metrics: []
validation: {}
sweep: {}
"""
        f = tmp_path / "config.yaml"
        f.write_text(yaml_content)
        cfg = SheepyConfig.from_file(f)
        assert cfg.experiment.project_name == "yaml_test"
        assert cfg.hparams.batch_size == 64

    def test_backward_compat_legacy_json(self, tmp_path):
        """Test loading the original sheepy JSON format."""
        legacy = {
            "experiment": {
                "project_name": "SMS_Spam_Example",
                "data_module": "sms_spam_data_module",
                "model": "transformer_classifier",
            },
            "metrics": {
                "multilabel": False,
                "tracked_metrics": [
                    {"name": "accuracy", "average": "micro"},
                    {"name": "f1", "average": "macro"},
                ],
            },
            "hparams": {
                "label": "label",
                "text": "text",
                "batch_size": 16,
                "encoder_model": "google/bert_uncased_L-2_H-128_A-2",
                "num_labels": 2,
            },
            "validation": {
                "metric": "val/AUROC",
                "metric_goal": "max",
            },
            "sweep": {
                "name": "sweep_v1",
                "method": "bayes",
                "metric": {"name": "val/AUROC", "goal": "maximize"},
                "parameters": {"learning_rate": {"min": 1e-5, "max": 5e-4}},
            },
        }
        f = tmp_path / "config.json"
        f.write_text(json.dumps(legacy))
        cfg = SheepyConfig.from_legacy_json(f)
        assert cfg.experiment.model == "transformer_classifier"
        assert cfg.hparams.num_labels == 2
        assert len(cfg.metrics.tracked_metrics) == 2


class TestHparamsConfig:
    def test_max_epochs_default(self):
        h = HparamsConfig(num_epochs=20)
        assert h.max_epochs == 20

    def test_max_epochs_override(self):
        h = HparamsConfig(num_epochs=20, max_epochs=5)
        assert h.max_epochs == 5

    def test_extra_fields_allowed(self):
        h = HparamsConfig(custom_field="hello")
        assert h.custom_field == "hello"
