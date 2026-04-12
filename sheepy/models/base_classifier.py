"""Base classifier built on Lightning 2.x with modern W&B logging.

Supports binary, multiclass, and multilabel classification through
a single ``task`` parameter that governs loss, metrics, and confusion matrices.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any

import lightning as L
import torch
import wandb

from sheepy.common.logger import get_logger
from sheepy.metrics.classification_metrics import ClassificationMetrics

logger = get_logger(__name__)


class BaseClassifier(L.LightningModule):
    """Abstract base for all sheepy classifiers.

    Args:
        config: Flat dict of experiment configuration (merged hparams + experiment + validation).
        data: LightningDataModule providing label_encoder, train_class_sizes, etc.
        task: ``"binary"``, ``"multiclass"``, or ``"multilabel"``.
    """

    def __init__(
        self,
        config: dict[str, Any],
        data: L.LightningDataModule,
        task: str | None = None,
    ) -> None:
        """Infer ``task`` from config when omitted and construct ``ClassificationMetrics``."""
        super().__init__()
        self.config = config
        self.data = data
        self.live_eval_mode = False

        if task is None:
            metrics_cfg = config.get("metrics", {})
            if metrics_cfg.get("multilabel", False):
                task = "multilabel"
            elif config.get("num_labels", 2) > 2:
                task = "multiclass"
            else:
                task = "binary"
        self.task = task

        self._build_metrics()

    def on_train_start(self) -> None:
        """Ensure the HF encoder stack is in train mode before the first optimization step."""
        # Lightning / HF: summary and hooks can leave encoder submodules in eval mode;
        # force full model train before the first training step.
        if hasattr(self, "model") and isinstance(self.model, torch.nn.Module):
            self.model.train()

    # ------------------------------------------------------------------
    # Metrics & Loss
    # ------------------------------------------------------------------
    def _build_metrics(self) -> None:
        """Attach a ``ClassificationMetrics`` instance sized to the label encoder vocab."""
        metrics_cfg = self.config.get("metrics", {})
        self.metrics = ClassificationMetrics(
            num_labels=len(self.data.label_encoder.vocab),
            config=metrics_cfg,
            task=self.task,
        )

    def _get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights from ``train_class_sizes`` for ``CrossEntropyLoss``."""
        weights = [0.0] * self.data.label_encoder.vocab_size
        total = sum(self.data.train_class_sizes.values())
        for class_name, count in self.data.train_class_sizes.items():
            ratio = count / total
            idx = self.data.label_encoder.encode(class_name).item()
            weights[idx] = 1.0 / ratio
        return torch.tensor(weights)

    def _build_loss(self) -> None:
        """Configure BCE (multilabel) or weighted cross-entropy from data module statistics."""
        if self.task == "multilabel":
            pos_weights = [self.data.pos_weights[k] for k in self.config["label"]]
            self._loss = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.FloatTensor(pos_weights)
            )
        else:
            self.class_weights = self._get_class_weights()
            self._loss = torch.nn.CrossEntropyLoss(weight=self.class_weights)

    def _build_model(self) -> None:
        """Subclasses must implement the underlying torch module(s)."""
        raise NotImplementedError("Subclasses must implement _build_model()")

    def configure_optimizers(self):
        """Subclasses must implement optimizers and optional schedulers."""
        raise NotImplementedError("Subclasses must implement configure_optimizers()")

    # ------------------------------------------------------------------
    # Forward & Loss
    # ------------------------------------------------------------------
    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        """Subclasses must return a dict including ``logits`` for the batch."""
        raise NotImplementedError("Subclasses must implement forward()")

    def loss(self, predictions: dict, targets: dict) -> torch.Tensor:
        """Compute task-appropriate loss from logits and dense labels."""
        logits = predictions["logits"]
        labels = targets["labels"]
        if self.task == "multilabel":
            return self._loss(logits.float(), labels.float())
        return self._loss(logits, labels)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch: tuple, batch_idx: int) -> dict:
        """Forward, loss, and per-step train metrics logging."""
        inputs, targets, _ = batch
        model_out = self.forward(**inputs)
        loss_train = self.loss(model_out, targets)

        logits = model_out["logits"]
        labels = targets["labels"]

        self.log("train/loss", loss_train, on_step=True, on_epoch=True, prog_bar=True)

        output_metrics = self.metrics.compute_metrics(logits, labels, stage="train")
        self.log_dict(output_metrics, on_step=True, on_epoch=True)

        return {"loss": loss_train, "logits": logits.detach(), "target": labels.detach()}

    def on_train_epoch_end(self) -> None:
        """Clear train metric accumulators between epochs."""
        self.metrics.reset("train")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validation_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        """Validation loss and metrics via shared eval path."""
        return self._shared_eval_step(batch, batch_idx, stage="val")

    def on_validation_epoch_end(self) -> None:
        """Reset validation metric state after the epoch."""
        self.metrics.reset("val")

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    def test_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        """Test loss and metrics via shared eval path."""
        return self._shared_eval_step(batch, batch_idx, stage="test")

    def on_test_epoch_end(self) -> None:
        """Reset test metric state after the epoch."""
        self.metrics.reset("test")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_step(self, batch: tuple, batch_idx: int, dataloader_idx: int = 0):
        """Return logits and sample ids for downstream CSV writing."""
        inputs, _, ids = batch
        model_out = self.forward(**inputs)
        return OrderedDict({
            "logits": model_out["logits"],
            "sample_id_keys": ids.get("sample_id_keys"),
        })

    # ------------------------------------------------------------------
    # Shared eval logic
    # ------------------------------------------------------------------
    def _shared_eval_step(self, batch: tuple, batch_idx: int, stage: str) -> OrderedDict:
        """Shared validation/test forward, loss, metric logging, and detached outputs."""
        inputs, targets, _ = batch
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)

        logits = model_out["logits"]
        labels = targets["labels"]

        self.log(f"{stage}/loss", loss_val, on_step=False, on_epoch=True, prog_bar=True)
        output_metrics = self.metrics.compute_metrics(logits, labels, stage=stage)
        self.log_dict(output_metrics, on_step=False, on_epoch=True)

        return OrderedDict({
            f"{stage}/loss": loss_val,
            "logits": logits.detach(),
            "target": labels.detach(),
        })

    # ------------------------------------------------------------------
    # Confusion matrix (W&B)
    # ------------------------------------------------------------------
    def _log_confusion_matrix(self, outputs: list[dict], stage: str) -> None:
        """Upload per-label or multiclass confusion matrices to the active W&B run."""
        logits = torch.cat([o["logits"] for o in outputs]).cpu()
        targets = torch.cat([o["target"] for o in outputs]).cpu()

        if self.task == "multilabel":
            for label_idx, label in enumerate(self.data.label_encoder.vocab):
                preds = (logits[:, label_idx] >= 0.0).int().numpy()
                trg = targets[:, label_idx].int().numpy()
                cm = wandb.plot.confusion_matrix(
                    y_true=trg.tolist(),
                    preds=preds.tolist(),
                    class_names=[f"Not_{label}", label],
                )
                if self.logger and hasattr(self.logger, "experiment"):
                    self.logger.experiment.log({f"{stage}/confusion_matrix/{label}": cm})
        else:
            preds = torch.argmax(logits, dim=1).numpy()
            trg = targets.numpy()
            cm = wandb.plot.confusion_matrix(
                y_true=trg.tolist(),
                preds=preds.tolist(),
                class_names=self.data.label_encoder.vocab,
            )
            if self.logger and hasattr(self.logger, "experiment"):
                self.logger.experiment.log({f"{stage}/confusion_matrix": cm})

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Also persist HuggingFace weights and tokenizer next to Lightning checkpoints."""
        if hasattr(self, "model") and hasattr(self.model, "save_pretrained"):
            save_path = os.path.join(self.config.get("output_dir", "."), "best_tfrm")
            self.model.config.update({"custom_args": self.config})
            self.model.save_pretrained(save_path)
            self.data.tokenizer.tokenizer.save_pretrained(save_path)
