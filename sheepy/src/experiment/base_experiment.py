import argparse
import json
import os
from argparse import ArgumentError, Namespace

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.cuda import device_count

from sheepy.src.common.logger import get_std_out_logger

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
            self.args.metrics = config["metrics"]

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
        self._build_trainer()

        self._prepare_logger()

    # TODO: merge this method with prepare_trainer()
    def prepare_evaluator(self, data_module_cls: pl.LightningDataModule, model_cls: pl.LightningModule):
        """Runs through all the module loading and checks to get the model ready to run evaluation.

        Args:
            data_module_cls (pl.LightningDataModule): Your data module class, which lives somewhere in src/data_modules
            model_cls (pl.LightningModule): pl.LightningModule, Your model module class, which lives somewhere in src/models
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

        if load_from_checkpoint:

            # Restore the arugments for the model and data module
            args_f = os.path.join(self.args.pretrained_dir, "args.json")
            with open(args_f, "r") as fp:
                args = json.load(fp)
                model_args = Namespace(**args)

            # Load the original data module to obtain the label encoder and class sizes
            model_data = data_module_cls(model_args)
            model_data.load_data_module()

            self.data.label_encoder = model_data.label_encoder
            self.data.train_class_sizes = model_data.train_class_sizes

            if hasattr(model_data, 'pos_weights'):
                self.data.pos_weights = model_data.pos_weights

            # Then the model from the checkpoint. Note that in pytl v1 the 'checkpoints' subdirectory was removed
            checkpoints = [os.path.join(self.args.pretrained_dir, f) for f in os.listdir(
                self.args.pretrained_dir) if f.endswith(".ckpt")]
            if len(checkpoints) != 1:
                raise ValueError("Your model directory {} should have exactly one .ckpt file in it. Instead, there are {}".format(
                    self.args.pretrained_dir, len(checkpoints)))

            # TODO - double check this -1 behavior on overfit model
            self.checkpoint_path = checkpoints[0]
            self.model = model_cls.load_from_checkpoint(self.checkpoint_path, args=model_args, data=self.data)

            self.model.live_eval_mode = self.args.evaluate_live

        else:
            self.model = model_cls(self.args, self.data)

        if hasattr(self.model, "text_representation"):
            self.logger.debug("Text Representation Encoder is frozen")
            self.model.text_representation.freeze_encoder()

        self._build_callbacks()
        self._build_trainer()
        self._prepare_logger()

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
            log_steps = min(self.args.validation['gradient_log_steps'], self.args.validation['param_log_steps'])
        elif self.args.validation['gradient_log_steps'] is None and self.args.validation['param_log_steps'] is not None:
            log_steps = self.args.validation['param_log_steps']
        elif self.args.validation['gradient_log_steps'] is not None and self.args.validation['param_log_steps'] is None:
            log_steps = self.args.validation['gradient_log_steps']
        else:
            log_steps = None
        # TODO - self.log_hyperparams and self.log_metrics should be called here too eventually
        self.wandb.watch(self.model, log="all", log_freq=log_steps)

    def _build_callbacks(self):
        """Helper function that reads model config arguments and builds out which callbacks need to run. A callback is
        something that runs after a step in training, such as a learning rate decay, or a checkpoint function, or some
        additional monitors, etc.

        If you want to add additional callbacks, search as self.grad_accum_callback, or self.lr_monitor_callback, then add them
        to self.custom_callbacks
        """
        self.checkpoint_callback = self._build_checkpoint_callback()
        self.lr_monitor_callback = LearningRateMonitor()
        self.custom_callbacks = [self.checkpoint_callback, self.lr_monitor_callback]
        if self.args.hparams['early_stop_enabled']:
            self.early_stop_callback = self._build_early_stop_callback()
            self.custom_callbacks.append(self.early_stop_callback)

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
            save_top_k=1,
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
            strategy="dp" if self.args.n_gpu >= 2 else None,
            precision=self.args.precision
        )

    def train(self):
        self.trainer.fit(self.model, self.data)

    def test(self):
        self.trainer.test(self.model, test_dataloaders=self.data.test_dataloader())

    def predict_batch(self):
        self.trainer.predict(self.model, dataloaders=self.data.predict_batch_dataloader())

    def predict_live(self):
        self.trainer.predict(self.model, dataloaders=self.data.predict_live_dataloader())

    @staticmethod
    def validate_experiment_args(args: argparse.Namespace):
        """Verifies that the experiment args passed are valid. For example, if the model is set to evaluation mode,
        we need a directory of the trained model, etc. Throws an error if anything is off.

        Args:
            args (argparse.Namespace): argument namespace to be checked and validated and augmented,

        Returns:
            args (argparse.Namespace): argument namespace after being modified.

        Raises:
            argparse.ArgumentError if something is amiss, or awry, or afoul....or askew.
        """
        args.n_gpu = device_count()
        args.precision = 16 if args.fp16 and args.n_gpu > 0 else 32

        if args.evaluate:

            if not args.evaluate_live:
                if not args.evaluate_batch_file:
                    raise ArgumentError("When running in batch evaluation mode, you must provide a text file to the evaluate_batch_file argument")
                if not os.path.exists(args.evaluate_batch_file):
                    raise ArgumentError("Cannot find evaluate_batch_file in file system located at {}".format(args.evaluate_batch_file))

            if args.experiment_name is None or args.output_key is None:
                raise ArgumentError(
                    "You must pass a pretrained experiment name and the output column name or dict key as output_key.")

            if args.pretrained_dir is None:
                args.pretrained_dir = os.path.join(
                    args.output_dir, args.project_name, args.experiment_name)

            args.output_dir = os.path.join(
                args.pretrained_dir, 'eval')

            args.output_prediction_path = os.path.join(args.output_dir, "batch_predictions.csv")

            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
        else:
            if args.experiment_name is None:
                args.experiment_name = "{}bit_{}_v{}".format(
                    args.precision, args.time, args.version)

            args.output_dir = os.path.join(
                args.output_dir, args.project_name, args.experiment_name)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        return args

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """ Return the argument pasrser with necessary args for this class appended to it """
        parser.add_argument("--project_name", type=str, default="test",
                            help="A custom project name. Will house all experiments under this project directory. First level under the output dir")
        parser.add_argument("--experiment_name", type=str, default=None,
                            help="A custom experiment name. Otherwise it will be be set to {16/32}bit_{time}_{version}, but give it a name/ Second level under the output dir")
        parser.add_argument("--version", type=str, default="0",
                            help="Version of model. Defaults to 0")
        parser.add_argument("--verbose", action="store_true",
                            help="Enable for debug-level logging")
        return parser
