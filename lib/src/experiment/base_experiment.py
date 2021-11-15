import argparse
import json
from typing import List, Dict
import pytorch_lightning as pl
import os
from argparse import Namespace
from lib.src.common.logger import get_std_out_logger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# TODO - add restore from checkpoint

class Experiment():
    def __init__(self, args: argparse.Namespace, logger=None, evaluation=False):
        self.evaluation = evaluation
        self.logger = logger if logger is not None else get_std_out_logger()
        if self.evaluation:
            self.logger.debug("Running in evaluation mode...")
        self.args = args
        self._validate_config()
        self._dump_args()
        # Access wandb functions via self.wandb.experiment.some_wandb_function()
        self._build_wandb_logger()

    def _validate_config(self):
        """
        Validate JSON configuration for the experiment and attach it to the args.
        """
        with open(self.args.config, "r") as fp:
            config = json.load(fp)
            for k, v in config["experiment"].items():
                setattr(self.args, k, v)

            self.args.sweep = config["sweep"]
            self.args.validation = config["validation"]
            self.args.hparams = config["hparams"]

    def prepare_trainer(self, data_module_cls: pl.LightningDataModule, model_cls: pl.LightningModule, pretrained_experiment_folder: str = None):
        """
        The prepare function is called on the experiment class to set up the two pytorch lightning abstractions used by the framework,
        namely, the data module, and the model module. According to other experiment args, various other things will get set up to track
        your experiment and log tshings.

        Args:
            data_module_cls - pl.LightningDataModule, Your data module class, which lives somewhere in src/data_modules
            model_cls - pl.LightningModule, Your model module class, which lives somewhere in src/models
            pretrained_experiment_folder - str, A folder that stores the pretrained experiment. Must exist if not None. 

        Returns:
            None
        """
        # First, we initialize the data. This is all the ETL
        self.data = data_module_cls(self.args)

        # PyTL exposes this method in case you want to call this early and not at runtime. In this case we do.
        self.data.prepare_data()  # This is called only once and on ONE gpu
        self.data.setup(stage="fit")  # This is called on every GPU separately
        self.data._verify_module()
        self.data.save_data_module()

        # Then we load up a model, and make sure it agrees with the data.
        self.model = model_cls(self.args, self.data)
        self._build_callbacks()
        self._build_trainer()  # TODO - checkpoint check here?

        self._prepare_logger()

    # TODO: merge this method with prepare_trainer()
    def prepare_evaluator(self, data_module_cls: pl.LightningDataModule, model_cls: pl.LightningModule):
        """
        Implements transfer learning
        Args:
            data_module_cls - pl.LightningDataModule, Your data module class, which lives somewhere in src/data_modules
            model_cls - pl.LightningModule, Your model module class, which lives somewhere in src/models
            will automatically look at args.output_dir

        Raises:
            ValueError: If pretrained experiment folder doesn't exist or doesn't have the necessary files (data module, checkpoints folder, wandb folder, args.json)

        """

        if self.args.pretrained_dir:
            load_from_checkpoint = True
            self.logger.info("Loading model from {}".format(
                self.args.pretrained_dir))
        else:
            load_from_checkpoint = False

        self.data = data_module_cls(self.args)
        self.data.prepare_data()
        self.data.setup(stage="test")

        if load_from_checkpoint:

            # Restore the arugments for the model and data module
            args_f = os.path.join(self.args.pretrained_dir, "args.json")
            with open(args_f, "r") as fp:
                args = json.load(fp)
                model_args = Namespace(**args)

            # Load the original data module to obtain the label encoder and class sizes
            model_data = data_module_cls(model_args)
            model_data.load_data_module()

            if hasattr(self.data, 'label_encoder'):
                # text2text models do not have label_encoder and train_class_sizes
                self.data.label_encoder = model_data.label_encoder
                self.data.train_class_sizes = model_data.train_class_sizes

            # Then the model from the checkpoint. Note that in pytl v1 the 'checkpoints' subdirectory was removed
            checkpoints = [os.path.join(self.args.pretrained_dir, f) for f in os.listdir(
                self.args.pretrained_dir) if f.endswith(".ckpt")]
            if not len(checkpoints):
                raise ValueError("Unable to locate at least one model checkpoint file in directory {}".format(
                    self.args.pretrained_dir))

            # TODO - double check this -1 behavior on overfit model
            self.checkpoint_path = checkpoints[-1]
            self.model = model_cls.load_from_checkpoint(
                self.checkpoint_path, args=model_args, data=self.data)

        else:
            self.model = model_cls(self.args, self.data)

        if hasattr(self.model, "text_representation"):
            self.logger.debug("Text Representation Encoder is frozen")
            self.model.text_representation.freeze_encoder()

        self._build_callbacks()
        self._build_trainer()
        self._prepare_logger()

    def evaluate_live(self):
        """Readability-wrapper as a call to the model's evaluate_live() method if your model supports a REPL-style evaluator"""
        self.model.evaluate_live(self.data)

    def evaluate_file(self, *args, **kwargs):
        """
        Readability-wrapper as a call to the model's evaluate() method for a single file.
        """
        self.model.evaluate_file(*args, **kwargs)

    def evaluate(self) -> List[dict]:
        """
        The main evaluate() call that calls a PyTorch Trainer. This expects a model that
        has been loaded from a checkpoint as well as a test dataloader that was prepared
        in the model itself.

        """
        self.trainer.test(model=self.model,
                          test_dataloaders=self.data.test_dataloader())

    def _dump_args(self):
        """Helper function that will dump all arguments into a model's directory
        so it can be loaded/examined at a later date"""
        args_path = os.path.join(self.args.output_dir, "args.json")
        with open(args_path, "w") as fp:
            args_dict = vars(self.args)
            json.dump(args_dict, fp, indent=4)

    def _build_wandb_logger(self):
        """Helper function to dump weights and biases files into a model's directory"""
        logger = WandbLogger(name=self.args.experiment_name,
                             save_dir=self.args.output_dir,
                             project=self.args.project_name,
                             version=self.args.version)
        self.wandb = logger

    def _prepare_logger(self):
        """Helper function that sets the verbosity of logging in weights and biases and pytorch"""
        if self.args.validation['gradient_log_steps'] is not None and self.args.validation['param_log_steps'] is not None:
            log = "all"
            log_steps = min(
                self.args.validation['gradient_log_steps'], self.args.validation['param_log_steps'])
            # TODO - add logger about them being different here
        elif self.args.validation['gradient_log_steps'] is None and self.args.validation['param_log_steps'] is not None:
            log = "parameters"
            log_steps = self.args.validation['param_log_steps']
        elif self.args.validation['gradient_log_steps'] is not None and self.args.validation['param_log_steps'] is None:
            log = "gradients"
            log_steps = self.args.validation['gradient_log_steps']
        else:
            log, log_steps = None, None
        # TODO - self.log_hyperparams and self.log_metrics should be called here too eventually
        self.wandb.watch(self.model, log=None, log_freq=log_steps)

    def _build_callbacks(self):
        """Helper function that reads model config arguments and builds out which callbacks need to run. A callback is
        something that runs after a step in training, such as a learning rate decay, or a checkpoint function, or some
        additional monitors, etc.

        If you want to add additional callbacks, search as self.grad_accum_callback, or self.lr_monitor_callback, then add them
        to self.custom_callbacks
        """
        self.checkpoint_callback = self._build_checkpoint_callback(
        ) if self.args.validation['save_top_k'] >= 1 else False

        self.early_stop_callback = self._build_early_stop_callback(
        ) if self.args.hparams['early_stop_enabled'] else False

        self.lr_monitor_callback = LearningRateMonitor()
        self.custom_callbacks = [self.checkpoint_callback]

    def _build_early_stop_callback(self):
        """Helper function to wrap around PyTL's early-stopping function, around validation loss only at this moment"""
        early_stop = EarlyStopping(
            monitor=self.args.validation['metric'],
            min_delta=0.,
            patience=3,
            verbose=False,
            mode=self.args.validation['metric_goal']
        )
        return early_stop

    def _build_checkpoint_callback(self):
        """Helper functioon to create the model checkpoints, and point PyTL where to overwrite them during training"""
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.args.output_dir,
            save_top_k=self.args.validation['save_top_k'],
            verbose=True,
            monitor=self.args.validation['metric'],
            mode=self.args.validation['metric_goal'],
        )
        return checkpoint_callback

    def _build_trainer(self):
        self.trainer = pl.Trainer(
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
            accelerator="dp" if self.args.n_gpu >= 2 else None,
            amp_level='O1',
            precision=self.args.precision
        )

    def train(self):
        self.trainer.fit(self.model, self.data)

    def predict(self, sample: dict):
        return self.model.predict(self.data, sample)

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """ Return the argument pasrser with necessary args for this class appended to it """
        parser.add_argument("--output_dir", type=str, required=True,
                            help="This should just be a path to the base experiment directory of your projects locally")
        parser.add_argument("--project_name", type=str, default="test",
                            help="A custom project name. Will house all experiments under this project directory. First level under the output dir")
        parser.add_argument("--experiment_name", type=str, default=None,
                            help="A custom experiment name. Otherwise it will be be set to {16/32}bit_{time}_{version}, but give it a name/ Second level under the output dir")
        parser.add_argument("--version", type=str, default="0",
                            help="Version of model. Defaults to 0")
        parser.add_argument("--verbose", action="store_true",
                            help="Enable for debug-level logging")
        return parser
