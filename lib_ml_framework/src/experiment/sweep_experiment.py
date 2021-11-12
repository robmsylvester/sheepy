import wandb
import pytorch_lightning as pl
from torch.multiprocessing import set_start_method
from argparse import Namespace
from lib_ml_framework.src.experiment.base_experiment import Experiment

# Note, to access params chosen from the sweep, refer to them as wandb.config, and this references
# the parameters object in the sweep from your JSON configuration file. Call wandb.init() first to
# gain access to this wandb.config dict.


class SweepExperiment(Experiment):
    def __init__(self, args: Namespace, data_module_cls: pl.LightningDataModule, model_cls: pl.LightningModule):
        super().__init__(args)
        self.data_module_cls = data_module_cls
        self.model_cls = model_cls
        set_start_method('spawn', force=True)  # multiprocessing necessity
        self.logger = None

    def _build_sweep_trainer(self):
        """A sweep trainer is really no different than a normal trainer module, except that we return
        it directly for each process thread via a return statement instead of an assignment, and the
        accelerator is specified as 'dp' due to the multiprocessing requiremets of the sweep"""
        return pl.Trainer(
            callbacks=self.custom_callbacks,
            logger=self.wandb,
            gradient_clip_val=self.args.hparams['gradient_clip_val'],
            gpus=self.args.n_gpu,
            log_gpu_memory="all",
            deterministic=True,
            check_val_every_n_epoch=self.args.validation['check_val_every_n_epoch'],
            fast_dev_run=False,
            accumulate_grad_batches=self.args.hparams['accumulate_grad_batches'],
            max_epochs=self.args.hparams['num_epochs'],
            val_check_interval=self.args.validation['val_check_interval'],
            accelerator='dp',
            amp_level='O1',
            precision=self.args.precision
        )

    def sweep_iteration(self):
        """
        The sweep iteration is a defined function to pass to the tuner, and in this case, it's just
        the normal prepare_trainer() function from the base experiment, without saving the module
        directly, and some modifications to remove logger's so that they can be serialized without
        threading errors.
        """
        data = self.data_module_cls(self.args)
        data.prepare_data()
        data.setup(stage="fit")
        data._verify_module()

        model = self.model_cls(self.args, data)
        self._build_callbacks()
        trainer = self._build_sweep_trainer()

        # python tries to pickle these in a subprocess, and they can't be pickled
        data.logger = None
        model.logger = None

        trainer.fit(model, data)

    def tune(self):
        """
        Runs a sweep or hyperparameters that can be visualized in the browser.
        """
        sweep_id = wandb.sweep(
            self.args.sweep, project=self.args.experiment_name)
        wandb.agent(sweep_id, function=self.sweep_iteration)
