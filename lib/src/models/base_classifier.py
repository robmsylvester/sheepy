import torch
import argparse
import pytorch_lightning as pl
import numpy as np
import wandb
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import LightningDataModule
from collections import OrderedDict
from lib.src.common.logger import get_std_out_logger
from lib.src.metrics.classification_metrics import ClassificationMetrics

# TODO - replace args with args or leave and get implicitly tracked by lightning. hmmmm


class BaseClassifier(pl.LightningModule):
    """
    Base model with wandb, logging, data support, ignoring many model implementation details.

    :param args: ArgumentParser containing the hyperparameters and other config
    :param data: LightningDataModule object containing implementations of train_dataloader, val_dataloader,
     and necessary other ETL.
    """

    def __init__(self, args: argparse.Namespace, data: LightningDataModule, logger=None) -> None:
        super().__init__()
        self.args = args
        self.data = data
        self.logger = logger if logger is not None else get_std_out_logger()
        self._build_metrics()
        self._set_tune_params()
        # self.save_hyperparameters()

    def _set_tune_params(self):
        # TODO - we may want to support iterating through other config than hparams, such as the encoder_model
        if self.args.tune:
            wandb.init()  # gives access to wandb.config
            for k, v in wandb.config.items():
                if k in self.args.hparams:
                    self.args.hparams[k] = v
                    self.logger.debug(
                        "Setting model hyperparameter {} to sweep value {}".format(k, v))

    def _build_model(self):
        raise NotImplementedError(
            "_build_model() has not been implemented. You must override this in your classifier")

    def _build_metrics(self) -> None:
        """Builds out the basic metrics for a classifier. This needs to be implemented in your classifier by
        instantiating the ClassificationMetrics class
        """
        self.metrics = ClassificationMetrics(
            self.data.label_encoder.vocab,
            self.args.metrics,
            logger=self.logger
        )

    def _get_class_weights(self):
        """
        By default, we scale weights of the classification by the ratio of the classes in the
        training set. 
        """
        weights = [None] * self.data.label_encoder.vocab_size

        # Normalize the ratios
        scale_factor = 1.0/sum(self.data.train_class_sizes.values())
        train_class_ratios = self.data.train_class_sizes
        for class_name, num_samples in self.data.train_class_sizes.items():
            train_class_ratios[class_name] = num_samples*scale_factor

        # and now invert each value to get its weight
        for class_name, ratio in train_class_ratios.items():
            label_index = self.data.label_encoder.encode(class_name)
            weights[label_index] = 1./ratio

        self.logger.info("\nClass weights:\n{}".format(weights))
        return torch.tensor(weights)

    def log_metrics(self, metrics: dict):
        """Helper function to track metrics by wandb"""
        if not isinstance(self.logger, WandbLogger):
            raise ValueError("self.logger is not a wandb logger")
        self.logger.log_metrics(metrics)

    def log_hyperparams(self):
        """Helper function to track hyperparameters by wandb"""
        if not isinstance(self.logger, WandbLogger):
            return

    def _build_loss(self):
        """ Initializes the loss function """
        raise NotImplementedError(
            "_build_loss() has not been implemented. You must override this in your classifier")

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        raise NotImplementedError(
            "configure_optimizers() has not been implemented. You must override this in your classifier")

    def predict(self, data_module: LightningDataModule, sample: dict) -> dict:
        """Evaluation function

        Args:
            data_module (LightningDataModule): module with method prepare_sample()
            sample (dict): Dictionary with correct key that specifies text column and value as text we want to classify

        Returns:
            dict: Dictionary with the input text and the predicted label.
        """
        pass

    def evaluate_live(self):
        """
        Expose a method that can call predictions live on sample data, text through a terminal for example.
        Essentially this is a barebones data prepare + predict REPL
        """
        raise NotImplementedError(
            "Your model does not implement an evaluate_live() method. See base_transformer_classifier.evaluate_live() for an example")

    def forward(self, tokens, lengths) -> dict:
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]
        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        raise NotImplementedError(
            "forward() has not been implemented. You must override this in your classifier")

    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]
        Returns:
            torch.tensor with loss value.
        """
        return self._loss(predictions["logits"], targets["labels"])

    def training_step(self, batch: tuple, batch_idx: int, *args, **kwargs) -> dict:
        """ 
        Runs one training step. This usually consists in the forward function followed
            by the loss function.

        :param batch: The output of your dataloader. 
        :param batch_nb: Integer displaying which batch this is
        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        inputs, targets, _ = batch
        model_out = self.forward(**inputs)
        loss_train = self.loss(model_out, targets)

        logits = model_out["logits"]
        labels = targets["labels"]

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_train = loss_train.unsqueeze(0)

        self.log('train/loss', loss_train, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        output = OrderedDict({
            "loss": loss_train, #we would normally want to name this 'train/loss', but pytorch lightning wants it to be 'loss'
            "logits": logits,
            "target": labels,
        })

        return output
    
    def training_step_end(self, outputs: dict):
        """Synchronizes metrics across GPU's in DP mode by updating and computing given the dictionary
        of outputs from the validation_step called on each GPU

        Args:
            outputs (dict): Return value of validation_step

        Returns:
            None
        """
        outputs['loss'] = outputs['loss'].sum() #To backpropagate, loss needs to be aggregated across GPUs
        output_metrics = self.metrics.compute_metrics(outputs['logits'], outputs['target'], stage='train')
        self.log_dict(output_metrics, on_step=True, on_epoch=True)
        return outputs

    def training_epoch_end(self, outputs: list) -> dict:
        """Runs pytorch lightning validation_epoch_end_function. For more details, see
        _run_epoch_end_metrics

        Args:
            outputs - list of dictionaries returned by step, across multiple gpus.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        cm = self._create_confusion_matrix(outputs, name="Train Epoch Confusion Matrix")
        self.log("train/epoch_confusion_matrix", cm, on_step=False, on_epoch=True)
        return None

    # TODO - this implementation is too specific for the base class. Move to children
    def validation_step(self, batch: tuple, batch_nb: int) -> OrderedDict:
        """[summary]

        Args:
            batch (tuple): [description]
            batch_nb (int): [description]

        Returns:
            OrderedDict: [description]
        """
        return self._shared_evaluation_step(batch, batch_nb, stage="val")
    
    def validation_step_end(self, outputs: dict):
        """Synchronizes metrics across GPU's in DP mode by updating and computing given the dictionary
        of outputs from the validation_step called on each GPU

        Args:
            outputs (dict): Return value of validation_step

        Returns:
            None
        """
        output_metrics = self.metrics.compute_metrics(outputs['logits'], outputs['target'], stage='val')
        self.log_dict(output_metrics, on_step=False, on_epoch=True)
        return outputs
    
    def validation_epoch_end(self, outputs: list) -> dict:
        """Runs pytorch lightning validation_epoch_end_function. For more details, see
        _run_epoch_end_metrics

        Args:
            outputs - list of dictionaries returned by step, across multiple gpus.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        cm = self._create_confusion_matrix(outputs, name="Validation Epoch Confusion Matrix")
        self.log("val/epoch_confusion_matrix", cm, on_step=False, on_epoch=True)
        return None
    
    def test_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        """[summary]

        Args:
            batch (tuple): [description]
            batch_idx (int): [description]

        Returns:
            OrderedDict: [description]
        """
        return self._shared_evaluation_step(batch, batch_idx, stage="test")

    def test_step_end(self, outputs: list) -> dict:
        """[summary]

        Args:
            outputs (list): [description]

        Returns:
            dict: [description]
        """

        # df = pd.DataFrame(
        #     list(map(int, output_results['sample_id_keys'].tolist())),
        #     columns=[self.data.sample_id_col]
        # )

        # softmax_p = F.softmax(output_results['logits'], dim=1).tolist()
        # df[self.data.args.output_key] = softmax_p

        # return df
        output_metrics = self.metrics.compute_metrics(outputs['logits'], outputs['target'], stage='test')
        self.log_dict(output_metrics, on_step=False, on_epoch=True)
        return outputs

    def test_epoch_end(self, outputs: list) -> dict:
        """Triggers self.data to write predictions to the disk"""
        # df_collected = pd.concat(outputs)
        # self.data.write_predictions_to_disk(df_collected)
        # return None
        cm = self._create_confusion_matrix(outputs, name="Test Epoch Confusion Matrix")
        self.log("test/epoch_confusion_matrix", cm, on_step=False, on_epoch=True)
        return None
    
    def _shared_evaluation_step(self, batch: tuple, batch_idx: int, stage: str) -> OrderedDict:
        """[summary]

        Args:
            batch (tuple): [description]
            batch_idx (int): [description]
            stage (str): [description]

        Returns:
            OrderedDict: [description]
        """
        inputs, targets, _ = batch
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)

        labels = targets["labels"]
        logits = model_out["logits"]

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        loss_key = stage + '/loss'
        output = OrderedDict({
            loss_key: loss_val,
            "logits": logits,
            "target": labels
        })

        return output

    
    def predict_step(self, batch: tuple, batch_idx: int, dataloader_idx: int=0):
        """
        PyTorch Lightning function called once per test step
        """
        inputs, _, ids = batch
        model_out = self.forward(**inputs)
        return {**model_out, **ids}

    def _create_confusion_matrix(self, outputs: list, name="Confusion Matrix"):
        """
        Given predictions and targets tensors, create a visual confusion matrix from matplotlib that can
        be loaded into weights and biases under the 'Media' tab. Use this pattern to add more visuals.

        Args:
            predictios: torch.tensor: the raw numeric predictions of the classifier.
            targets: torch.tensor: the raw numeric target labels of the classifier.
        """

        logits = torch.cat([output['logits']
                          for output in outputs]).detach().cpu()
        pred = torch.argmax(logits, dim=1)
        trg = torch.cat([output['target']
                            for output in outputs]).detach().cpu()

        predictions = self.data.label_encoder.batch_decode(pred)
        target = self.data.label_encoder.batch_decode(trg)

        confmatrix = confusion_matrix(
            predictions, target, labels=self.data.label_encoder.vocab)
        confdiag = np.eye(len(confmatrix)) * confmatrix
        np.fill_diagonal(confmatrix, 0)

        confmatrix = confmatrix.astype('float')
        n_confused = np.sum(confmatrix)
        confmatrix[confmatrix == 0] = np.nan
        confmatrix = go.Heatmap({'coloraxis': 'coloraxis1', 'x': self.data.label_encoder.vocab, 'y': self.data.label_encoder.vocab, 'z': confmatrix,
                                 'hoverongaps': False, 'hovertemplate': 'Predicted %{y}<br>instead of %{x}<br>on %{z} examples<extra></extra>'})

        confdiag = confdiag.astype('float')
        n_right = np.sum(confdiag)
        confdiag[confdiag == 0] = np.nan
        confdiag = go.Heatmap({'coloraxis': 'coloraxis2', 'x': self.data.label_encoder.vocab, 'y': self.data.label_encoder.vocab, 'z': confdiag,
                               'hoverongaps': False, 'hovertemplate': 'Predicted %{y} correctly<br>on %{z} examples<extra></extra>'})

        fig = go.Figure((confdiag, confmatrix))
        transparent = 'rgba(0, 0, 0, 0)'
        n_total = n_right + n_confused
        fig.update_layout({'coloraxis1': {'colorscale': [[0, transparent], [0, 'rgba(180, 0, 0, 0.05)'], [
                          1, f'rgba(180, 0, 0, {max(0.2, (n_confused/n_total) ** 0.5)})']], 'showscale': False}})
        fig.update_layout({'coloraxis2': {'colorscale': [[0, transparent], [
                          0, f'rgba(0, 180, 0, {min(0.8, (n_right/n_total) ** 2)})'], [1, 'rgba(0, 180, 0, 1)']], 'showscale': False}})

        xaxis = {'title': {'text': 'y_true'}, 'showticklabels': False}
        yaxis = {'title': {'text': 'y_pred'}, 'showticklabels': False}

        fig.update_layout(title={'text': name, 'x': 0.5},
                          paper_bgcolor=transparent, plot_bgcolor=transparent, xaxis=xaxis, yaxis=yaxis)

        return wandb.data_types.Plotly(fig)

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """ Return the argument parser with necessary args for this class appended to it """
        return parser
