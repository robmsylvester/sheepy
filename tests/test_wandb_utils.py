"""Tests for sheepy.wandb_utils."""
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


class TestBuildPredictionTable:
    def test_basic(self):
        from sheepy.wandb_utils.tables import build_prediction_table

        table = build_prediction_table(
            texts=["hello", "world"],
            true_labels=["a", "b"],
            predicted_labels=["a", "a"],
            confidences=[0.9, 0.6],
        )
        assert len(table.data) == 2
        assert table.columns == ["sample_id", "text", "true_label", "predicted_label", "confidence", "correct"]
        assert table.data[0][-1] is True  # first prediction correct
        assert table.data[1][-1] is False  # second wrong


class TestBuildDatasetSummaryTable:
    def test_basic(self):
        from sheepy.wandb_utils.tables import build_dataset_summary_table

        table = build_dataset_summary_table({"spam": 100, "ham": 400})
        assert len(table.data) == 2


class TestPerClassMetrics:
    def test_basic(self):
        from sheepy.wandb_utils.tables import build_per_class_metrics_dict

        result = build_per_class_metrics_dict(
            class_names=["a", "b", "c"],
            metric_name="F1",
            values=[0.8, 0.9, 0.7],
            prefix="val",
        )
        assert result["val/F1/a"] == pytest.approx(0.8)
        assert result["val/F1/c"] == pytest.approx(0.7)

    def test_with_tensor(self):
        from sheepy.wandb_utils.tables import build_per_class_metrics_dict

        result = build_per_class_metrics_dict(
            class_names=["x", "y"],
            metric_name="Recall",
            values=torch.tensor([0.5, 0.6]),
        )
        assert result["val/Recall/x"] == pytest.approx(0.5)


class TestExperimentLogger:
    def test_log_experiment_metadata(self):
        from sheepy.wandb_utils.experiment_logger import ExperimentLogger

        mock_run = MagicMock()
        mock_run.config = {}
        el = ExperimentLogger(mock_run, ["a", "b"])
        el.log_experiment_metadata({"key": "value"})
        assert "git_commit" in mock_run.config
        assert "config_hash" in mock_run.config

    def test_update_summary(self):
        from sheepy.wandb_utils.experiment_logger import ExperimentLogger

        mock_run = MagicMock()
        mock_run.summary = {}
        el = ExperimentLogger(mock_run, ["a", "b"])
        el.update_summary("best_f1", 0.95)
        assert mock_run.summary["best_f1"] == 0.95

    def test_log_confusion_matrix_string_labels_calls_wandb_with_ints(self, monkeypatch):
        """wandb.plot.confusion_matrix expects int indices when class_names is set."""
        from sheepy.wandb_utils import experiment_logger as el_mod
        from sheepy.wandb_utils.experiment_logger import ExperimentLogger

        captured: dict = {}

        def fake_confusion_matrix(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(el_mod.wandb.plot, "confusion_matrix", fake_confusion_matrix)

        mock_run = MagicMock()
        el = ExperimentLogger(mock_run, ["spam", "ham"])
        el.log_confusion_matrix(["ham", "spam", "ham"], ["spam", "ham", "ham"], stage="test")

        assert captured["y_true"] == [1, 0, 1]
        assert captured["preds"] == [0, 1, 1]
        assert captured["class_names"] == ["spam", "ham"]
        mock_run.log.assert_called_once()

    def test_labels_to_indices_invalid_string_raises(self):
        from sheepy.wandb_utils.experiment_logger import _labels_to_class_indices

        with pytest.raises(ValueError, match="not in class_names"):
            _labels_to_class_indices(["ok", "bogus"], ["ok", "no"])

    def test_log_dataset_summary_skips_duplicate(self):
        from sheepy.wandb_utils.experiment_logger import ExperimentLogger

        mock_run = MagicMock()
        mock_run.config = {"_sheepy_dataset_summary_logged": True}
        el = ExperimentLogger(mock_run, ["a", "b"])
        el.log_dataset_summary({"a": 1, "b": 2})
        mock_run.log.assert_not_called()
