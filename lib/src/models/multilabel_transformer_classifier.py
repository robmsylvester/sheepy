import torch
import argparse
import wandb
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict
from pytorch_lightning import LightningDataModule
from lib.src.models.base_transformer_classifier import TransformerClassifier
from collections import OrderedDict

#TODO - inherit from transformer as well as multilabel base
class MultiLabelTransformerClassifier(TransformerClassifier):
    """
    Sample model to show how to use a Transformer model to classify sentences.

    This model builds upon the text representation class then attaches a head from the fully connected classifier class,
    stitching them together according to your config params

    :param args: ArgumentParser containing the hyperparameters.
    :param data: LightningDataModule object containing implementations of train_dataloader, val_dataloader,
     and necessary other ETL.
    """

    def __init__(self, args: argparse.Namespace, data: LightningDataModule) -> None:
        super().__init__(args, data)

    def _get_class_weights(self):
        """
        For the multilabel loss function, we still have sparse labels, it's just they are treated
        independently. So in this case, we just have a list of values, and for each value, there
        is a weight for the positives that is different. We will pass this list of positives to
        the BCEWithLogitsLoss function.

        For more info, see:
        https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        """
        return self.data.pos_weights

    def _build_loss(self):
        """ Initializes the loss function/s."""
        # have to do it this way to preserve label order
        pos_weights = [self.data.pos_weights[k]
                       for k in self.args.hparams["label"]]
        pos_weights = torch.FloatTensor(pos_weights)
        self._loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # TODO - maybe eliminate this. seems not to be called
    # TODO: allow batch prediction?
    def predict(self, data_module: LightningDataModule, sample: dict) -> dict:
        """ Predict function.
        Args:
            data_module: module with method prepare_sample()
            Sample: Dictionary with correct key that specifies text column and value as text we want to classify
        Returns:
            Dictionary with the input text and the predicted labels
        """

        if self.training:
            self.eval()

        with torch.no_grad():
            model_input, _ = data_module.prepare_sample([sample])
            model_out = self.forward(**model_input)
            logits = model_out["logits"].numpy()

            predictions = (logits >= 0.5).astype(int) #TODO - arg out this default 0.5

            predicted_labels = {label: predictions[label_idx] for label_idx, label in enumerate(
                self.args.hparams['label'])}
            sample["predicted_labels"] = predicted_labels

        return sample

    def predict_prob(self, data_module: LightningDataModule, sample: dict) -> dict:
        """
        Predict function that returns probability

        Args:
            data_module: module with method prepare_sample()
            sample: Dictionary with correct key that specifies text column and value as text we want to classify
        Returns:
            Dictionary with the input text and the predicted softmax label probability
        """

        if self.training:
            self.eval()

        with torch.no_grad():
            model_input, _ = data_module.prepare_sample([sample])
            model_out = self.forward(**model_input)
            logits = model_out["logits"].numpy()
            return logits

    # TODO - verify these validation, test, predict handlers arent needed to be overridden here and then remove them 1 by 1.
    # the ones that do need to be overridden can go in the multilabel base classifier
    def validation_step(self, batch: tuple, batch_nb: int) -> OrderedDict:
        """[summary]

        Args:
            batch (tuple): [description]
            batch_nb (int): [description]

        Returns:
            OrderedDict: [description]
        """
        return self._shared_evaluation_step(batch, batch_nb, stage="val")

    def training_epoch_end(self, outputs: List) -> dict:
        """Runs pytorch lightning validation_epoch_end_function. For more details, see
        _run_epoch_end_metrics

        Args:
            outputs - list of dictionaries returned by step, across multiple gpus.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        cms = self._create_confusion_matrices(outputs)
        for label_name, confusion_matrix in cms.items():
            self.logger.experiment.log({"train/epoch_confusion_matrix/{}".format(label_name): confusion_matrix})
        return None
    
    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        """Runs pytorch lightning validation_epoch_end_function. For more details, see
        _run_epoch_end_metrics

        Args:
            outputs - list of dictionaries returned by step, across multiple gpus.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        cms = self._create_confusion_matrices(outputs)
        for label_name, confusion_matrix in cms.items():
            self.logger.experiment.log({"val/epoch_confusion_matrix/{}".format(label_name): confusion_matrix})
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

    def test_epoch_end(self, outputs: List[Dict]):
        """[summary]

        Args:
            outputs (List[Dict]): [description]

        Returns:
            [type]: [description]
        """
        cms = self._create_confusion_matrices(outputs)
        for label_name, confusion_matrix in cms.items():
            self.logger.experiment.log({"test/epoch_confusion_matrix/{}".format(label_name): confusion_matrix})
        return None

    def predict_step(self, batch: tuple, batch_idx: int, dataloader_idx: int=0):
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

        sample_ids = ids['sample_id_keys']

        output = OrderedDict({
            "logits": logits,
            "sample_id_keys": sample_ids,
        })

        return output
    
    #NOTE - PyTorch Lightning 1.5.1 still uses this on_ prefix for predict_step_end, but this may change soon. see here: https://github.com/PyTorchLightning/pytorch-lightning/issues/9380
    def on_predict_step_end(self, outputs: List[Dict]) -> List:
        return outputs
    
    #NOTE - PyTorch Lightning 1.5.1 still uses this on_ prefix for predict_epoch_end, but this may change soon. see here: https://github.com/PyTorchLightning/pytorch-lightning/issues/9380
    def on_predict_epoch_end(self, outputs: List) -> Dict:
        if self.live_eval_mode:
            prediction_logits = outputs[0][0]["logits"].cpu().squeeze()
            prediction_softmax = torch.nn.Softmax(dim=0)(prediction_logits)
            output_str = "\nPrediction:\n"
            for label_idx, label in enumerate(self.data.label_encoder.vocab):
                output_str += "\t{}:{}\n".format(label, prediction_softmax[label_idx])
            self.logger.info(output_str)
        else:
            self.data._write_predictions(outputs[0])
        return None


    #This probably becomes the shared eval step
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

        # We want to get an overall picture of the loss but also track each label individually.
        # Overall loss can be tracked on the step
        loss_val = self.loss(model_out, targets)
        labels = targets["labels"]
        logits = model_out["logits"]
        preds = (logits >= 0.5).float()  # should use a threshold argument
        val_acc = torch.sum(labels == preds).item() / (len(labels) * 1.0)
        val_acc = torch.tensor(val_acc)

        loss_key = stage + '/loss'
        acc_key = stage + '/acc'

        output = OrderedDict({
            loss_key: loss_val,
            acc_key: val_acc,
            "logits": logits,
            "pred": preds,
            "target": labels
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output
    
    def _create_confusion_matrices(self, outputs: Dict) -> Dict:
        """[summary]

        Args:
            outputs (Dict): [description]

        Returns:
            Dict: [description]
        """
        cms = {}
        for label_idx, label in enumerate(self.data.label_encoder.vocab):
            logits = torch.cat([output['logits'][:, label_idx]
                            for output in outputs]).detach().cpu().numpy()
            pred = (logits >= 0.5).astype(int) #TODO - arg out this default 0.5
            trg = torch.cat([output['target'][:, label_idx]
                                for output in outputs]).detach().cpu().numpy()

            cm = wandb.plot.confusion_matrix(y_true=trg, preds=pred, class_names=["Not_{}".format(label), label])
            cms[label] = cm
        return cms

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """ Return the argument parser with necessary args for this class appended to it """
        return parser