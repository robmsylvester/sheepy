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
from pytorch_lightning.metrics.functional.classification import auroc, multiclass_auroc
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import OrderedDict
from lib.src.common.logger import get_std_out_logger

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
        """
        This function establishes the metrics that we want to track in weights and biases for the experiment
        TODO - these should be part of the experiment profile, not hardcoded
        """
        self.metrics = {
            'f1': f1_score,
            'precision': precision_score,
            'auroc': auroc if self.args.hparams['num_labels'] <= 2 else multiclass_auroc,
            'recall': recall_score
        }

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

    # TODO: allow batches?
    def predict(self, data_module: LightningDataModule, sample: dict) -> dict:
        """ Predict function.
        :param sample: dictionary with the text we want to classify.
        Returns:
            Dictionary with the input text and the predicted label.
        """
        if self.training:
            self.eval()
        with torch.no_grad():
            raise NotImplementedError(
                "predict() has not been implemented. You must override this in your classifier")

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

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
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

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_train = loss_train.unsqueeze(0)
            #loss_val = torch.mean(loss_val)

        tqdm_dict = {"train_loss": loss_train}
        output = OrderedDict(
            {"loss": loss_train, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )

        # can also return just a scalar instead of a dict (return loss_val)
        # return output

        self.log('loss', loss_train, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss_train

    # TODO - this implementation is too specific for the base class. Move to children
    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.

        Args:
            batch - tuple of the features and labels
            batch_nb - integer specifying the batch iteration number

        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, targets, _ = batch
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)

        y = targets["labels"]
        y_hat = model_out["logits"]

        # some classification metrics
        # TODO - migrate these over to additional metrics
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        output = OrderedDict({
            "val_loss": loss_val,
            "val_acc": val_acc,
            "pred": labels_hat,
            "target": y
        })

        # can also return just a scalar instead of a dict (return loss_val)

        return output

    # TODO - this implementation is too specific for the base class. Move to children
    def validation_epoch_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.

        A lot of this logic here defines what to do in the case of multiple GPU's. In this case,
        what we do with the validation loss is average the number out across the GPU's.

        For other metrics, we can detach all of the predictions and targets from the GPU, concatenate them,
        and just calculate them as one large vector.

        Args:
            outputs - list of dictionaries returned by validation step, across multiple gpus.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        val_loss_mean = 0
        val_acc_mean = 0

        pred = torch.cat([output['pred']
                          for output in outputs]).detach().cpu()
        target = torch.cat([output['target']
                            for output in outputs]).detach().cpu()
        
        one_hot_shape = (pred.shape[0], len(self.data.label_encoder.vocab))
        one_hot_pred = np.zeros(one_hot_shape)
        rows = np.arange(pred.shape[0])
        one_hot_pred[rows, pred] = 1
        one_hot_pred = torch.tensor(one_hot_pred, dtype=torch.float32)

        # Set certain metric outputs to 0 if there are no positives.
        total_positives = torch.sum(target).cpu().numpy(
        ) if self.on_gpu else torch.sum(target).numpy()

        if total_positives == 0:
            wandb.termwarn(
                "Warning, no sample targets were found that are positive. Setting certain metrics to output 0 for this validation iteration")
            auroc = f1 = precision = recall = torch.tensor([0.])
        else:
            
            average_param = "micro" if len(self.data.label_encoder.vocab) == 2 else None
            label_param = None if len(self.data.label_encoder.vocab) == 2 else [self.data.label_encoder.encode(k) for k in self.data.label_encoder.vocab]

            f1 = self.metrics['f1'](
                target.numpy(), pred.numpy(), pos_label=1, labels=label_param, average=average_param)

            auroc = self.metrics['auroc'](pred, target) if len(self.data.label_encoder.vocab) == 2 else self.metrics['auroc'](one_hot_pred, target)
            precision = self.metrics['precision'](target.numpy(), pred.numpy(), pos_label=1, labels=label_param, average=average_param)
            recall = self.metrics['recall'](target.numpy(), pred.numpy(), pos_label=1, labels=label_param, average=average_param)

        # The confusion matrix we can construct always, regardless of whether or not we have any positives
        cm = self._create_confusion_matrix(pred, target)

        # We will use the mean loss across all of the GPU's as the loss
        for output in outputs:
            val_loss = output["val_loss"]

            # reduce manually when using dp or ddp2
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output["val_acc"]
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}
        result = {  # TODO - this is probably useless.
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss_mean,
            'f1': f1,
            'auroc': auroc,
            'precision': precision,
            'recall': recall
        }
        tracked_metrics = {
            "val_loss": val_loss_mean,
            "val_acc": val_acc_mean,
            'f1': f1,
            'auroc': auroc,
            'precision': precision,
            'recall': recall,
            'cm': cm
        }

        self.log_metrics(tracked_metrics)
        self.log("recall", recall, prog_bar=True)
        self.log("precision", precision, prog_bar=True)
        self.log("f1", f1, prog_bar=True)
        self.log("auroc", auroc, prog_bar=True)
        self.log("val_loss", val_loss_mean, prog_bar=True)
        self.log("val_acc", val_acc_mean, prog_bar=True)
        return None

    def test_step(self, batch: tuple, batch_idx: int):
        """
        PyTorch Lightning function called once per test step
        """
        inputs, _, ids = batch  # we don't care about the targets
        model_out = self.forward(**inputs)
        return {**model_out, **ids}

    def test_step_end(self, output_results):
        df = pd.DataFrame(
            list(map(int, output_results['sample_id_keys'].tolist())),
            columns=[self.data.sample_id_col]
        )

        softmax_p = F.softmax(output_results['logits'], dim=1).tolist()
        df[self.data.args.output_key] = softmax_p

        return df

    def test_epoch_end(self, outputs):
        """Triggers self.data to write predictions to the disk"""
        df_collected = pd.concat(outputs)
        self.data.write_predictions_to_disk(df_collected)
        return None

    def _create_confusion_matrix(self, predictions: torch.tensor, target: torch.tensor):
        """
        Given predictions and targets tensors, create a visual confusion matrix from matplotlib that can
        be loaded into weights and biases under the 'Media' tab. Use this pattern to add more visuals.

        Args:
            predictios: torch.tensor: the raw numeric predictions of the classifier.
            targets: torch.tensor: the raw numeric target labels of the classifier.
        """
        predictions = self.data.label_encoder.batch_decode(predictions)
        target = self.data.label_encoder.batch_decode(target)

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

        fig.update_layout(title={'text': 'Confusion Matrix', 'x': 0.5},
                          paper_bgcolor=transparent, plot_bgcolor=transparent, xaxis=xaxis, yaxis=yaxis)

        return wandb.data_types.Plotly(fig)

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """ Return the argument parser with necessary args for this class appended to it """
        return parser
