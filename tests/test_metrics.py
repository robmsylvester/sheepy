"""Tests for sheepy.metrics.ClassificationMetrics."""
import pytest
import torch

from sheepy.metrics.classification_metrics import ClassificationMetrics


@pytest.fixture
def binary_config():
    return {
        "multilabel": False,
        "tracked_metrics": [
            {"name": "accuracy", "average": "micro"},
            {"name": "f1", "average": "macro"},
            {"name": "precision", "average": "macro"},
            {"name": "recall", "average": "macro"},
        ],
    }


@pytest.fixture
def multiclass_config():
    return {
        "multilabel": False,
        "tracked_metrics": [
            {"name": "accuracy", "average": "micro"},
            {"name": "f1", "average": "macro"},
        ],
    }


@pytest.fixture
def multilabel_config():
    return {
        "multilabel": True,
        "tracked_metrics": [
            {"name": "accuracy", "average": "micro"},
            {"name": "f1", "average": "macro"},
        ],
    }


class TestBinaryMetrics:
    def test_compute_returns_dict(self, binary_config):
        m = ClassificationMetrics(num_labels=2, config=binary_config, task="binary")
        logits = torch.tensor([[2.0, 0.1], [0.1, 3.0], [1.5, 0.5]])
        labels = torch.tensor([0, 1, 0])
        result = m.compute_metrics(logits, labels, stage="val")
        assert isinstance(result, dict)
        assert any("Accuracy" in k or "accuracy" in k.lower() for k in result)

    def test_all_correct(self, binary_config):
        m = ClassificationMetrics(num_labels=2, config=binary_config, task="binary")
        logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
        labels = torch.tensor([0, 1])
        result = m.compute_metrics(logits, labels, stage="val")
        acc_key = [k for k in result if "accuracy" in k.lower()][0]
        assert result[acc_key].item() == pytest.approx(1.0)

    def test_reset(self, binary_config):
        m = ClassificationMetrics(num_labels=2, config=binary_config)
        m.reset("val")  # should not raise


class TestMulticlassMetrics:
    def test_compute(self, multiclass_config):
        m = ClassificationMetrics(num_labels=4, config=multiclass_config, task="multiclass")
        logits = torch.randn(8, 4)
        labels = torch.randint(0, 4, (8,))
        result = m.compute_metrics(logits, labels, stage="train")
        assert isinstance(result, dict)


class TestMultilabelMetrics:
    def test_compute(self, multilabel_config):
        m = ClassificationMetrics(num_labels=3, config=multilabel_config, task="multilabel")
        logits = torch.randn(4, 3)
        labels = torch.randint(0, 2, (4, 3)).float()
        result = m.compute_metrics(logits, labels, stage="test")
        assert isinstance(result, dict)


class TestInvalidStage:
    def test_raises_on_unknown_stage(self, binary_config):
        m = ClassificationMetrics(num_labels=2, config=binary_config)
        with pytest.raises(ValueError, match="Unknown stage"):
            m.compute_metrics(torch.randn(2, 2), torch.tensor([0, 1]), stage="invalid")


class TestEmptyTrackedMetrics:
    def test_defaults_to_accuracy(self):
        m = ClassificationMetrics(num_labels=2, config={"tracked_metrics": []}, task="binary")
        result = m.compute_metrics(torch.randn(2, 2), torch.tensor([0, 1]), stage="val")
        assert len(result) > 0
