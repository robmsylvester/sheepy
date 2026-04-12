"""Tests for sheepy.experiment."""
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from sheepy.config.schema import SheepyConfig
from sheepy.experiment.base_experiment import Experiment


class TestExperimentConfig:
    def test_validate_training_config(self, tmp_path):
        config = {
            "evaluate": False,
            "fp16": False,
            "output_dir": str(tmp_path),
            "project_name": "test",
            "experiment_name": "test_run",
            "version": "0",
        }
        result = Experiment.validate_experiment_config(config)
        assert "n_gpu" in result
        assert os.path.exists(result["output_dir"])

    def test_validate_eval_config_no_batch_file_raises(self):
        config = {
            "evaluate": True,
            "evaluate_live": False,
            "evaluate_batch_file": None,
            "output_dir": ".",
        }
        with pytest.raises(ValueError, match="evaluate_batch_file"):
            Experiment.validate_experiment_config(config)

    def test_auto_experiment_name(self, tmp_path):
        config = {
            "evaluate": False,
            "fp16": False,
            "output_dir": str(tmp_path),
            "project_name": "test",
            "experiment_name": None,
            "version": "0",
        }
        result = Experiment.validate_experiment_config(config)
        assert result["experiment_name"] is not None


class TestExperimentInit:
    @patch("sheepy.experiment.base_experiment.WandbLogger")
    def test_init_from_sheepy_config(self, mock_wandb, tmp_path):
        cfg = SheepyConfig()
        flat = cfg.to_flat_dict()
        flat["output_dir"] = str(tmp_path)
        flat["experiment_name"] = "test"
        exp = Experiment(flat)
        assert exp.config["project_name"] == "default_project"
        assert os.path.exists(tmp_path / "config.json")

    @patch("sheepy.experiment.base_experiment.WandbLogger")
    def test_dump_config(self, mock_wandb, tmp_path):
        cfg = SheepyConfig()
        flat = cfg.to_flat_dict()
        flat["output_dir"] = str(tmp_path)
        flat["experiment_name"] = "test"
        exp = Experiment(flat)
        config_path = tmp_path / "config.json"
        assert config_path.exists()
        with open(config_path) as f:
            saved = json.load(f)
        assert saved["project_name"] == "default_project"
