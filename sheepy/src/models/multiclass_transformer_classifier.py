from argparse import Namespace

import torch
import wandb
from pytorch_lightning import LightningDataModule
from sklearn.metrics import f1_score, precision_score, recall_score

from sheepy.src.models.base_transformer_classifier import TransformerClassifier


class MulticlassTransformerClassifier(TransformerClassifier):
    """
    Similar to the base (binary) classifier, with adjustments made for multiclass datasets
    """

    def __init__(self, args: Namespace, data: LightningDataModule, logger=None) -> None:
        super().__init__(args, data)
        assert self.args.hparams['num_labels'] > 2, "The Config object sees num_labels expected to be 2. The multiclass classifier must use at least 3. If you have just two labels, use a base classifier"

    def _build_metrics(self) -> None:
        """
        This function establishes the metrics that we want to track in weights and biases for the experiment.

        Note this is just a small variation from the binary classifier where we kill off auroc
        """
        self.metrics = {
            'f1': f1_score,
            'precision': precision_score,
            'recall': recall_score
        }

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

        # Set certain metric outputs to 0 if there are no positives.
        total_positives = torch.sum(target).cpu().numpy(
        ) if self.on_gpu else torch.sum(target).numpy()

        if total_positives == 0:
            wandb.termwarn(
                "Warning, no sample targets were found that are positive. Setting certain metrics to output 0.")
            # 1 or 0 is standard. 0 is nice though because then you go up from the beginning :)
            f1 = precision = recall = torch.tensor([0.])
        else:
            f1 = self.metrics['f1'](target.numpy(), pred.numpy())
            precision = self.metrics['precision'](target.numpy(), pred.numpy())
            recall = self.metrics['recall'](target.numpy(), pred.numpy())

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
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss_mean,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        tracked_metrics = {
            "val_loss": val_loss_mean,
            "val_acc": val_acc_mean,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'cm': cm
        }

        self.log_metrics(tracked_metrics)
        self.log("recall", recall, prog_bar=True)
        self.log("precision", precision, prog_bar=True)
        self.log("f1", f1, prog_bar=True)
        self.log("val_loss", val_loss_mean, prog_bar=True)
        self.log("val_acc", val_acc_mean, prog_bar=True)

        return None
