import argparse
import math
import os
from collections import OrderedDict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import wandb
from numpy import vectorize
from pytorch_lightning import LightningDataModule

from sheepy.common.logger import get_std_out_logger
from sheepy.models.base_classifier import BaseClassifier

logger = get_std_out_logger()


class MultiLabelBaseClassifier(BaseClassifier):
    def __init__(self, args: argparse.Namespace, data: LightningDataModule):
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
        """Initializes the loss function/s."""
        # have to do it this way to preserve label order
        pos_weights = [self.data.pos_weights[k] for k in self.args.label]
        pos_weights = torch.FloatTensor(pos_weights)

        # BCEWithLogitsLoss combines the sigmoid operation in a numerically stable fashion. that's why we don't include the
        # sigmoid in the model, and call it explicitly a couple times in the prediction functions
        self._loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]
        Returns:
            torch.tensor with loss value.
        """
        return self._loss(predictions["logits"].float(), targets["labels"].float())

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
            self.logger.experiment.log(
                {"train/epoch_confusion_matrix/{}".format(label_name): confusion_matrix}
            )

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        """Runs pytorch lightning validation_epoch_end_function. For more details, see
        _run_epoch_end_metrics

        Args:
            outputs - list of dictionaries returned by step, across multiple gpus.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.
        """
        self.plot_explanations(outputs)

        cms = self._create_confusion_matrices(outputs)
        for label_name, confusion_matrix in cms.items():
            self.logger.experiment.log(
                {"val/epoch_confusion_matrix/{}".format(label_name): confusion_matrix}
            )

    def plot_explanations(self, outputs):
        def model_predict_function(x):
            inputs = self.data.tokenizer.tokenizer(
                x.tolist(), return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)[0]
            scores = torch.nn.Softmax(dim=-1)(outputs)
            return torch.logit(scores).detach().cpu().numpy()

        explainer = shap.Explainer(
            model_predict_function, self.data.tokenizer.tokenizer, output_names=self.args.label
        )
        texts = [output["text"] for output in outputs]
        # Flatten the list of lists
        texts = [item for sublist in texts for item in sublist]
        logger.info(f"Generating shap values for {len(texts)} samples ...")
        shap_values = explainer(texts)
        for label in shap_values.output_names:
            shap.plots.bar(
                shap_values[:, :, label].mean(0), show=False, order=shap.Explanation.argsort.flip
            )
            self.logger.log_image(
                key=f"val/shap/{label}",
                images=[plt.gcf()],
            )
            plt.close()

    def test_epoch_end(self, outputs: List[Dict]):
        """[summary]

        Args:
            outputs (List[Dict]): [description]

        Returns:
            [type]: [description]
        """
        cms = self._create_confusion_matrices(outputs)
        for label_name, confusion_matrix in cms.items():
            self.logger.experiment.log(
                {"test/epoch_confusion_matrix/{}".format(label_name): confusion_matrix}
            )

    # NOTE - PyTorch Lightning 1.5.1 still uses this on_ prefix for predict_epoch_end, but this may change soon. see here: https://github.com/PyTorchLightning/pytorch-lightning/issues/9380
    def on_predict_epoch_end(self, outputs: List) -> Dict:
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        # define vectorized sigmoid
        sigmoid_v = vectorize(sigmoid)

        if self.live_eval_mode:
            prediction_logits = outputs[0][0]["logits"].cpu().squeeze()
            sigmoid_logits = sigmoid_v(prediction_logits)
            output_str = "\nPrediction:\n"
            for label_idx, label in enumerate(self.data.label_encoder.vocab):
                output_str += "\t{}:{}\n".format(label, sigmoid_logits[label_idx])
            print(output_str)
        else:
            self.data._write_predictions(outputs[0])

    # This probably becomes the shared eval step
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
        preds = (logits >= 0.0).float()  # should use a threshold argument
        val_acc = torch.sum(labels == preds).item() / (len(labels) * 1.0)
        val_acc = torch.tensor(val_acc)

        loss_key = stage + "/loss"
        acc_key = stage + "/acc"

        output = OrderedDict(
            {
                loss_key: loss_val,
                acc_key: val_acc,
                "logits": logits,
                "pred": preds,
                "target": labels,
                "text": self.data.tokenizer.tokenizer.batch_decode(
                    inputs["tokens"],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ),
            }
        )

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def write_outputs(self, texts, y_score, labels, metadata=None, mode: str = "test"):
        score_df = pd.DataFrame(
            y_score, columns=[f"score_{label_name}" for label_name in self.label_names]
        )
        true_df = pd.DataFrame(
            labels, columns=[f"true_{label_name}" for label_name in self.label_names]
        )

        text_df = pd.DataFrame({"text": texts, "metadata": metadata})
        df = pd.concat((text_df, score_df, true_df), axis=1)
        df = df.sample(min(10000, df.shape[0]))
        logger.info(f"Saving samples ... \n {df}")
        df.to_csv(os.path.join(self.hparams.output_dir, f"{mode}_sample.csv"), index=False)

    def _create_confusion_matrices(self, outputs: Dict) -> Dict:
        """[summary]

        Args:
            outputs (Dict): [description]

        Returns:
            Dict: [description]
        """
        cms = {}
        for label_idx, label in enumerate(self.data.label_encoder.vocab):
            logits = (
                torch.cat([output["logits"][:, label_idx] for output in outputs])
                .detach()
                .cpu()
                .numpy()
            )
            pred = (logits >= 0.0).astype(int)  # TODO - arg out this default
            trg = (
                torch.cat([output["target"][:, label_idx] for output in outputs])
                .detach()
                .cpu()
                .numpy()
            )

            cm = wandb.plot.confusion_matrix(
                y_true=trg, preds=pred, class_names=["Not_{}".format(label), label]
            )
            cms[label] = cm
        return cms
