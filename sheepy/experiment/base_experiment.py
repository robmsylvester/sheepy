"""Experiment orchestrator built on Lightning 2.x and modern W&B.

Manages the lifecycle of data preparation, model construction, training,
evaluation, and prediction.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger

from sheepy.common.compute_env import (
    resolve_trainer_accelerator_and_devices,
    strategy_for_distributed,
    summarize_compute_environment,
)
from sheepy.common.logger import get_logger
from sheepy.common.run_progress import run_note, run_phase_end, run_phase_start
from sheepy.common.warning_filters import apply_sheepy_training_warning_filters
from sheepy.config.schema import SheepyConfig

logger = get_logger(__name__)


class Experiment:
    """Orchestrates a sheepy training or evaluation run.

    Args:
        config: Either a ``SheepyConfig`` or a flat dict produced by ``SheepyConfig.to_flat_dict()``.
        evaluation: Whether we're in evaluation (inference) mode.
    """

    def __init__(
        self,
        config: SheepyConfig | dict[str, Any],
        evaluation: bool = False,
    ) -> None:
        """Normalize config, create output dir, dump JSON, and attach a W&B logger."""
        self.evaluation = evaluation

        if isinstance(config, SheepyConfig):
            self.sheepy_config = config
            self.config = config.to_flat_dict()
        else:
            self.sheepy_config = None
            self.config = config

        apply_sheepy_training_warning_filters()
        self._ensure_output_dir()
        self._dump_config()
        self._build_wandb_logger()
        _ce = self.config["compute_environment"]
        run_note(
            f"PyTorch {_ce['torch_version']}  |  CUDA available={_ce['torch_cuda_available']}  "
            f"devices={_ce['torch_cuda_device_count']}  "
            f"{(_ce['torch_cuda_device_names'][0] + ' …') if _ce['torch_cuda_device_names'] else ''}  "
            f"|  MPS={_ce['torch_mps_available']}"
        )
        if not _ce["torch_cuda_available"]:
            run_note(
                "No CUDA visible to PyTorch (often a CPU-only torch wheel). "
                "NVIDIA GPU: install CUDA-enabled torch from https://pytorch.org/get-started/locally/"
            )
        run_note(
            f"Experiment initialized  project={self.config.get('project_name')}  "
            f"name={self.config.get('experiment_name')}  "
            f"output_dir={self.config.get('output_dir')}"
        )
        logger.info(
            "experiment_initialized",
            project=self.config.get("project_name"),
            experiment_name=self.config.get("experiment_name"),
            output_dir=self.config.get("output_dir"),
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _ensure_output_dir(self) -> None:
        """Create ``output_dir`` if missing."""
        out = self.config.get("output_dir", ".")
        os.makedirs(out, exist_ok=True)

    def _dump_config(self) -> None:
        """Write the flat config (including ``compute_environment``) to ``config.json``."""
        out_path = os.path.join(self.config.get("output_dir", "."), "config.json")
        self.config["compute_environment"] = summarize_compute_environment()
        with open(out_path, "w") as fp:
            json.dump(self.config, fp, indent=2, default=str)

    def _build_wandb_logger(self) -> None:
        """Construct ``WandbLogger`` and drop deprecated ``anonymous=None`` init args when present."""
        self.wandb = WandbLogger(
            name=self.config.get("experiment_name", "sheepy_run"),
            save_dir=self.config.get("output_dir", "."),
            project=self.config.get("project_name", "sheepy"),
            version=str(self.config.get("version", "0")),
        )
        # W&B 0.25+: passing ``anonymous=None`` into ``wandb.init`` still triggers a deprecation
        # warning when the user is logged in. Lightning stores None explicitly — drop the key.
        init_args = getattr(self.wandb, "_wandb_init", None)
        if isinstance(init_args, dict) and init_args.get("anonymous") is None:
            init_args.pop("anonymous", None)

    def _prepare_logger(self) -> None:
        """Optionally ``wandb.watch`` the model for gradients/weights at the tightest log interval."""
        grad_steps = self.config.get("gradient_log_steps")
        param_steps = self.config.get("param_log_steps")
        steps = [s for s in (grad_steps, param_steps) if s is not None]
        log_freq = min(steps) if steps else None
        if log_freq:
            self.wandb.watch(self.model, log="all", log_freq=log_freq)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _build_callbacks(self) -> list:
        """Checkpoint, LR monitor, rich bar, and optional early stopping from config."""
        callbacks = [
            ModelCheckpoint(
                dirpath=self.config.get("output_dir", "."),
                save_top_k=1,
                verbose=True,
                monitor=self.config.get("metric", "val/loss"),
                mode=self.config.get("metric_goal", "min"),
            ),
            LearningRateMonitor(),
            RichProgressBar(),
        ]
        if self.config.get("early_stop_enabled", False):
            callbacks.append(
                EarlyStopping(
                    monitor=self.config.get("metric", "val/loss"),
                    min_delta=0.0,
                    patience=3,
                    verbose=False,
                    mode=self.config.get("metric_goal", "min"),
                )
            )
        self.custom_callbacks = callbacks

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    def _build_trainer(self) -> None:
        """Resolve accelerator/devices, build ``L.Trainer``, and refresh the on-disk config dump."""
        n_gpu = torch.cuda.device_count()
        precision = "16-mixed" if self.config.get("precision") == 16 and n_gpu > 0 else "32-true"

        accelerator, devices = resolve_trainer_accelerator_and_devices(self.config)
        strategy = strategy_for_distributed(accelerator, devices)
        self.config["lightning_accelerator"] = accelerator
        self.config["lightning_devices"] = devices
        self.config["lightning_strategy"] = strategy
        run_note(
            f"Lightning Trainer: accelerator={accelerator!r}  devices={devices!r}  "
            f"strategy={strategy!r}  precision={precision}"
        )

        self.trainer = L.Trainer(
            callbacks=self.custom_callbacks,
            logger=self.wandb,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
            deterministic=True,
            max_epochs=self.config.get("max_epochs", self.config.get("num_epochs", 10)),
            accumulate_grad_batches=self.config.get("accumulate_grad_batches", 1),
            gradient_clip_val=self.config.get("gradient_clip_val", 0),
            check_val_every_n_epoch=self.config.get("check_val_every_n_epoch", 1),
            val_check_interval=self.config.get("val_check_interval", 1.0),
        )
        # Persist resolved Lightning hardware next to the initial config dump.
        self._dump_config()

    # ------------------------------------------------------------------
    # Prepare + train + test
    # ------------------------------------------------------------------
    def prepare_trainer(
        self,
        data_module_cls: type[L.LightningDataModule],
        model_cls: type[L.LightningModule],
    ) -> None:
        """Instantiate data and model, verify/save the data module, callbacks, and trainer for training."""
        run_phase_start(
            "prepare_trainer",
            f"data_module={data_module_cls.__name__}  model={model_cls.__name__}",
        )
        run_note("Instantiating data module and running prepare_data() / setup(fit)")
        self.data = data_module_cls(self.config)
        self.data.prepare_data()
        self.data.setup(stage="fit")
        n_train = (
            len(self.data._train_dataset)
            if getattr(self.data, "_train_dataset", None) is not None
            else 0
        )
        run_phase_end("data_module_setup", f"train_dataset_size={n_train}")
        self.data._verify_module()
        self.data.save_data_module()
        run_note("Saved data module metadata to disk")

        run_phase_start("build_model", model_cls.__name__)
        self.model = model_cls(self.config, self.data)
        run_phase_end("build_model", "weights loaded / classifier head ready")

        run_note("Building Lightning callbacks (checkpoint, LR monitor, progress bar)")
        self._build_callbacks()
        run_note("Building Lightning Trainer (accelerator, epochs, precision)")
        self._build_trainer()
        self._prepare_logger()
        run_phase_end(
            "prepare_trainer",
            f"max_epochs={self.trainer.max_epochs}  monitor_metric={self.config.get('metric')}",
        )

    def prepare_evaluator(
        self,
        data_module_cls: type[L.LightningDataModule],
        model_cls: type[L.LightningModule],
    ) -> None:
        """Load checkpoint + saved data metadata when ``pretrained_dir`` is set; else build a fresh model."""
        run_phase_start("prepare_evaluator", f"data_module={data_module_cls.__name__}")
        pretrained_dir = self.config.get("pretrained_dir")
        self.data = data_module_cls(self.config)

        if pretrained_dir:
            run_note(f"Loading checkpoint bundle from {pretrained_dir}")
            logger.info("loading_pretrained", path=pretrained_dir)
            config_path = os.path.join(pretrained_dir, "config.json")
            with open(config_path, "r") as fp:
                saved_config = json.load(fp)

            model_data = data_module_cls(saved_config)
            model_data.load_data_module()
            self.data.label_encoder = model_data.label_encoder
            self.data.train_class_sizes = model_data.train_class_sizes
            if hasattr(model_data, "pos_weights"):
                self.data.pos_weights = model_data.pos_weights

            checkpoints = [
                os.path.join(pretrained_dir, f)
                for f in os.listdir(pretrained_dir)
                if f.endswith(".ckpt")
            ]
            if len(checkpoints) != 1:
                raise ValueError(
                    f"Expected exactly one .ckpt in '{pretrained_dir}', found {len(checkpoints)}."
                )

            self.model = model_cls.load_from_checkpoint(
                checkpoints[0], config=saved_config, data=self.data
            )
            self.model.live_eval_mode = self.config.get("evaluate_live", False)
        else:
            self.model = model_cls(self.config, self.data)

        self._build_callbacks()
        self._build_trainer()
        self._prepare_logger()
        run_phase_end("prepare_evaluator", "trainer ready for eval / predict")

    def train(self) -> None:
        """Run ``trainer.fit`` on the prepared model and data module."""
        run_phase_start(
            "trainer.fit",
            f"epochs={self.trainer.max_epochs}  (Lightning progress bar follows)",
        )
        self.trainer.fit(self.model, self.data)
        run_phase_end("trainer.fit", "training finished")

    def test(self) -> None:
        """Run the test loop once via Lightning."""
        run_phase_start("trainer.test", "running test_dataloader()")
        self.trainer.test(self.model, dataloaders=self.data.test_dataloader())
        run_phase_end("trainer.test", "test epoch finished")

    def predict_batch(self) -> None:
        """Run ``predict`` over the batch file dataloader."""
        self.trainer.predict(self.model, dataloaders=self.data.predict_batch_dataloader())

    def predict_live(self) -> None:
        """Run one ``predict`` pass on stdin-sourced samples."""
        self.trainer.predict(self.model, dataloaders=self.data.predict_live_dataloader())

    # ------------------------------------------------------------------
    # Static validation
    # ------------------------------------------------------------------
    @staticmethod
    def validate_experiment_config(config: dict[str, Any]) -> dict[str, Any]:
        """Validate and augment the config dict with computed fields."""
        config["n_gpu"] = torch.cuda.device_count()
        config["precision"] = 16 if config.get("fp16") and config["n_gpu"] > 0 else 32

        if config.get("evaluate"):
            if not config.get("evaluate_live") and not config.get("evaluate_batch_file"):
                raise ValueError("Batch evaluation requires 'evaluate_batch_file'.")

            if config.get("pretrained_dir") is None:
                config["pretrained_dir"] = os.path.join(
                    config.get("output_dir", "."),
                    config.get("project_name", ""),
                    "models",
                    config.get("experiment_name", ""),
                )
            config["output_dir"] = os.path.join(config["pretrained_dir"], "eval")
            config["output_prediction_path"] = os.path.join(config["output_dir"], "batch_predictions.csv")
        else:
            if not config.get("experiment_name"):
                config["experiment_name"] = f"{config['precision']}bit_v{config.get('version', 0)}"
            config["output_dir"] = os.path.join(
                config.get("output_dir", "."),
                config.get("project_name", ""),
                "models",
                config["experiment_name"],
            )

        os.makedirs(config["output_dir"], exist_ok=True)
        return config
