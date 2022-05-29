import argparse
import os
from collections import OrderedDict
from typing import Any, Dict, List

import pytorch_lightning as pl
import shap
import torch
import transformers
import wandb
from pytorch_lightning import LightningDataModule

from sheepy.metrics.classification_metrics import ClassificationMetrics


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
        self.live_eval_mode = False
        self._build_metrics()
        self._set_tune_params()

    def _set_tune_params(self):
        # TODO - we may want to support iterating through other config than hparams, such as the encoder_model
        if self.args.tune:
            wandb.init()  # gives access to wandb.config
            for k, v in wandb.config.items():
                if k in self.args.hparams:
                    self.args.hparams[k] = v
                    print("Setting model hyperparameter {} to sweep value {}".format(k, v))

    def _build_model(self):
        raise NotImplementedError(
            "_build_model() has not been implemented. You must override this in your classifier"
        )

    def _build_metrics(self) -> None:
        """Builds out the basic metrics for a classifier. This needs to be implemented in your classifier by
        instantiating the ClassificationMetrics class
        """
        self.metrics = ClassificationMetrics(self.data.label_encoder.vocab, self.args.metrics)

    def _get_class_weights(self):
        """
        By default, we scale weights of the classification by the ratio of the classes in the
        training set.
        """
        weights = [None] * self.data.label_encoder.vocab_size

        # Normalize the ratios
        scale_factor = 1.0 / sum(self.data.train_class_sizes.values())
        train_class_ratios = self.data.train_class_sizes
        for class_name, num_samples in self.data.train_class_sizes.items():
            train_class_ratios[class_name] = num_samples * scale_factor

        # and now invert each value to get its weight
        for class_name, ratio in train_class_ratios.items():
            label_index = self.data.label_encoder.encode(class_name)
            weights[label_index] = 1.0 / ratio

        print("\nClass weights:\n{}".format(weights))
        return torch.tensor(weights)

    def _build_loss(self):
        """Initializes the loss function/s."""
        self.class_weights = self._get_class_weights()
        self._loss = torch.nn.CrossEntropyLoss(weight=self.class_weights)

    def configure_optimizers(self):
        """Sets different Learning rates for different parameter groups."""
        raise NotImplementedError(
            "configure_optimizers() has not been implemented. You must override this in your classifier"
        )

    def forward(self, tokens, lengths) -> dict:
        """Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]
        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        raise NotImplementedError(
            "forward() has not been implemented. You must override this in your classifier"
        )

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

        self.log("train/loss", loss_train, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        output = OrderedDict(
            {
                "loss": loss_train,  # we would normally want to name this 'train/loss', but pytorch lightning wants it to be 'loss'
                "logits": logits,
                "target": labels,
            }
        )

        return output

    def training_step_end(self, outputs: dict):
        """Synchronizes metrics across GPU's in DP mode by updating and computing given the dictionary
        of outputs from the validation_step called on each GPU

        Args:
            outputs (dict): Return value of validation_step

        Returns:
            None
        """
        outputs["loss"] = outputs[
            "loss"
        ].sum()  # To backpropagate, loss needs to be aggregated across GPUs
        output_metrics = self.metrics.compute_metrics(
            outputs["logits"], outputs["target"], stage="train"
        )
        self.log_dict(output_metrics, on_step=True, on_epoch=True)
        return outputs

    def training_epoch_end(self, outputs: List) -> dict:
        """Runs pytorch lightning validation_epoch_end_function. For more details, see
        _run_epoch_end_metrics

        Args:
            outputs - list of dictionaries returned by step, across multiple gpus.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.
        """
        cm = self._create_confusion_matrix(outputs, name="Train Epoch Confusion Matrix")
        self.logger.experiment.log({"train/epoch_confusion_matrix": cm})

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
        output_metrics = self.metrics.compute_metrics(
            outputs["logits"], outputs["target"], stage="val"
        )
        self.log_dict(output_metrics, on_step=False, on_epoch=True)
        return outputs

    def validation_epoch_end(self, outputs: List) -> dict:
        """Runs pytorch lightning validation_epoch_end_function. For more details, see
        _run_epoch_end_metrics

        Args:
            outputs - list of dictionaries returned by step, across multiple gpus.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.
        """
        pipeline = transformers.pipline(
            "text-classification",
            model=self.model,
            tokenizer=self.data.tokenizer,
            return_all_scores=True,
        )
        explainer = shap.Explainer(pipeline)
        shap_values = explainer(outputs["text"])

        cm = self._create_confusion_matrix(outputs, name="Validation Epoch Confusion Matrix")
        self.logger.experiment.log({"val/epoch_confusion_matrix": cm})

    def test_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        """[summary]

        Args:
            batch (tuple): [description]
            batch_idx (int): [description]

        Returns:
            OrderedDict: [description]
        """
        return self._shared_evaluation_step(batch, batch_idx, stage="test")

    def test_step_end(self, outputs: List) -> dict:
        """[summary]

        Args:
            outputs (list): [description]

        Returns:
            dict: [description]
        """

        output_metrics = self.metrics.compute_metrics(
            outputs["logits"], outputs["target"], stage="test"
        )
        self.log_dict(output_metrics, on_step=False, on_epoch=True)
        return outputs

    def test_epoch_end(self, outputs: List[dict]) -> dict:
        """Triggers self.data to write predictions to the disk"""
        cm = self._create_confusion_matrix(outputs, name="Test Epoch Confusion Matrix")
        self.logger.experiment.log({"test/epoch_confusion_matrix": cm})

    def predict_step(self, batch: tuple, batch_idx: int, dataloader_idx: int = 0):
        """PyTorch Lightning function to do raw batch prediction

        Args:
            batch (tuple): [description]
            batch_idx (int): [description]
            dataloader_idx (int, optional): [description]. Defaults to 0.

        Returns:
            [type]: [description]
        """
        inputs, _, ids = batch
        model_out = self.forward(**inputs)
        logits = model_out["logits"]

        sample_ids = ids["sample_id_keys"]

        output = OrderedDict(
            {
                "logits": logits,
                "sample_id_keys": sample_ids,
            }
        )

        return output

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = os.path.join(self.args.output_dir, "best_tfrm")
        self.model.config.update(self.args.hparams)
        self.model.save_pretrained(save_path)
        self.data.tokenizer.tokenizer.save_pretrained(save_path)

    # NOTE - PyTorch Lightning 1.5.1 still uses this on_ prefix for predict_step_end, but this may change soon. see here: https://github.com/PyTorchLightning/pytorch-lightning/issues/9380
    def on_predict_step_end(self, outputs: list) -> list:
        return outputs

    # NOTE - PyTorch Lightning 1.5.1 still uses this on_ prefix for predict_epoch_end, but this may change soon. see here: https://github.com/PyTorchLightning/pytorch-lightning/issues/9380
    def on_predict_epoch_end(self, outputs: List) -> dict:
        if self.live_eval_mode:
            prediction_logits = outputs[0][0]["logits"].cpu().squeeze()
            prediction_softmax = torch.nn.Softmax(dim=0)(prediction_logits)
            output_str = "\nPrediction:\n"
            for label_idx, label in enumerate(self.data.label_encoder.vocab):
                output_str += "\t{}:{}\n".format(label, prediction_softmax[label_idx])
            print(output_str)
        else:
            self.data._write_predictions(outputs[0])

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

        loss_key = stage + "/loss"
        output = OrderedDict(
            {
                loss_key: loss_val,
                "logits": logits,
                "target": labels,
                "text": self.data.tokenizer.decode(inputs["input_ids"]),
            }
        )

        return output

    # TODO - move this function to classification metrics?
    def _create_confusion_matrix(self, outputs: list, name="Confusion Matrix"):
        logits = torch.cat([output["logits"] for output in outputs]).detach().cpu()
        pred = torch.argmax(logits, dim=1).numpy()
        trg = torch.cat([output["target"] for output in outputs]).detach().cpu().numpy()

        cm = wandb.plot.confusion_matrix(
            y_true=pred, preds=trg, class_names=self.data.label_encoder.vocab
        )
        return cm

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Return the argument parser with necessary args for this class appended to it"""
        return parser
