"""Tests for sheepy model classes."""
from unittest.mock import MagicMock

import pytest
import torch

from sheepy.models.base_classifier import BaseClassifier
from sheepy.nlp.label_encoder import LabelEncoder


def _make_mock_data(labels=None, task="binary"):
    labels = labels or ["spam", "ham"]
    data = MagicMock()
    data.label_encoder = LabelEncoder(labels)
    data.train_class_sizes = {l: 50 for l in labels}
    data.tokenizer = MagicMock()
    if task == "multilabel":
        data.pos_weights = {l: 2.0 for l in labels}
    return data


def _make_config(task="binary", labels=None):
    labels = labels or ["spam", "ham"]
    return {
        "encoder_model": "google/bert_uncased_L-2_H-128_A-2",
        "learning_rate": 3e-5,
        "encoder_learning_rate": 1e-5,
        "num_labels": len(labels),
        "label": labels if task == "multilabel" else "label",
        "metrics": {
            "multilabel": task == "multilabel",
            "tracked_metrics": [{"name": "accuracy", "average": "micro"}],
        },
    }


class TestBaseClassifier:
    def test_cannot_instantiate_abstract(self):
        config = _make_config()
        data = _make_mock_data()
        model = BaseClassifier(config, data, task="binary")
        with pytest.raises(NotImplementedError):
            model.forward(torch.zeros(1, 10), torch.tensor([10]))

    def test_loss_binary(self):
        config = _make_config()
        data = _make_mock_data()
        model = BaseClassifier(config, data, task="binary")
        model._build_loss()
        preds = {"logits": torch.randn(4, 2)}
        targets = {"labels": torch.randint(0, 2, (4,))}
        loss = model.loss(preds, targets)
        assert loss.dim() == 0  # scalar

    def test_loss_multilabel(self):
        labels = ["toxic", "obscene", "insult"]
        config = _make_config(task="multilabel", labels=labels)
        data = _make_mock_data(labels=labels, task="multilabel")
        model = BaseClassifier(config, data, task="multilabel")
        model._build_loss()
        preds = {"logits": torch.randn(4, 3)}
        targets = {"labels": torch.randint(0, 2, (4, 3)).float()}
        loss = model.loss(preds, targets)
        assert loss.dim() == 0

    def test_task_auto_detection_multilabel(self):
        labels = ["a", "b"]
        config = _make_config(task="multilabel", labels=labels)
        data = _make_mock_data(labels=labels, task="multilabel")
        model = BaseClassifier(config, data)
        assert model.task == "multilabel"

    def test_task_auto_detection_multiclass(self):
        labels = ["a", "b", "c"]
        config = _make_config(labels=labels)
        config["num_labels"] = 3
        config["metrics"]["multilabel"] = False
        data = _make_mock_data(labels=labels)
        model = BaseClassifier(config, data)
        assert model.task == "multiclass"


class TestTransformerClassifier:
    @pytest.mark.slow
    def test_forward_shape(self):
        from sheepy.models.base_transformer_classifier import TransformerClassifier

        config = _make_config()
        data = _make_mock_data()
        model = TransformerClassifier(config, data, task="binary")

        tokens = torch.randint(0, 1000, (2, 20))
        lengths = torch.tensor([20, 15])
        out = model.forward(tokens, lengths)
        assert "logits" in out
        assert out["logits"].shape == (2, 2)

    @pytest.mark.slow
    def test_configure_optimizers(self):
        from sheepy.models.base_transformer_classifier import TransformerClassifier

        config = _make_config()
        data = _make_mock_data()
        model = TransformerClassifier(config, data)
        opts, _ = model.configure_optimizers()
        assert len(opts) == 1
