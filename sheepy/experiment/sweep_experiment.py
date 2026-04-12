"""W&B sweep experiment orchestrator."""
from __future__ import annotations

from typing import Any

import lightning as L
import torch
import wandb

from sheepy.common.compute_env import (
    resolve_trainer_accelerator_and_devices,
    strategy_for_distributed,
)
from sheepy.common.logger import get_logger
from sheepy.experiment.base_experiment import Experiment

logger = get_logger(__name__)


class SweepExperiment(Experiment):
    """Runs W&B hyperparameter sweeps.

    Args:
        config: Flat experiment config dict.
        data_module_cls: Class for the data module.
        model_cls: Class for the model.
    """

    def __init__(
        self,
        config: dict[str, Any],
        data_module_cls: type[L.LightningDataModule],
        model_cls: type[L.LightningModule],
    ) -> None:
        """Store sweep participant classes after base experiment setup."""
        super().__init__(config)
        self.data_module_cls = data_module_cls
        self.model_cls = model_cls

    def _build_sweep_trainer(self) -> L.Trainer:
        """Lightweight trainer for a single W&B sweep trial (no progress callback list mutation)."""
        n_gpu = torch.cuda.device_count()
        precision = "16-mixed" if self.config.get("precision") == 16 and n_gpu > 0 else "32-true"

        accelerator, devices = resolve_trainer_accelerator_and_devices(self.config)
        strategy = strategy_for_distributed(accelerator, devices)

        return L.Trainer(
            callbacks=self.custom_callbacks,
            logger=self.wandb,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
            deterministic=True,
            max_epochs=self.config.get("max_epochs", self.config.get("num_epochs", 10)),
        )

    def sweep_iteration(self) -> None:
        """One agent run: merge sweep hyperparams, rebuild data/model, and ``fit``."""
        wandb.init()
        sweep_config = dict(wandb.config)
        run_config = {**self.config, **sweep_config}

        data = self.data_module_cls(run_config)
        data.prepare_data()
        data.setup(stage="fit")
        data._verify_module()

        model = self.model_cls(run_config, data)
        self._build_callbacks()
        trainer = self._build_sweep_trainer()
        trainer.fit(model, data)

    def tune(self) -> None:
        """Register a W&B sweep and start agents calling ``sweep_iteration``."""
        sweep_cfg = self.config.get("sweep", {})
        sweep_id = wandb.sweep(sweep_cfg, project=self.config.get("project_name", "sheepy"))
        wandb.agent(sweep_id, function=self.sweep_iteration)
