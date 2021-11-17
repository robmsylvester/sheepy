import torch
from typing import List, Any
from lib.src.common.logger import get_std_out_logger
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, AUROC, F1

class ClassificationMetrics():
    def __init__(self,
        labels: List[Any],
        pos_label=1,
        multilabel=False,
        separate_multiclass_labels=False,
        logger=None):
        """A class to help abstract the classification metrics that are reported and sent to PyTorch Lightning's
        validation epoch callbacks as well as Weights and Biases visualizations.

        Args:
            labels (List[Any]): List of all labels in your dataset.
            pos_label (int, optional): If binary, the label that is considered positive. Ignored if len(labels) > 2. Defaults to 1
            multilabel (bool, optional): If True, will create metrics for a multiclass, multilabel classifier. Defaults to False.
            separate_multiclass_labels (bool, optional): If True, will separate metrics for multiclass labels (f1, precision, recall). 
                Otherwise will combine them weighted by the number of samples in the dataset. Defaults to False.
            logger ([type], optional): Logger to console. Defaults to None to get standard logger
        """
    
        self.logger = get_std_out_logger() if logger is None else logger
        self.labels = labels
        self.num_labels=len(labels)
        self.pos_label=pos_label
        self.multilabel=multilabel
        self.separate_multiclass_labels=separate_multiclass_labels
        self.classification_threshold = 0.5 #move to config profile

        if self.num_labels < 2:
            raise ValueError("Metrics are ill-defined for fewer than two labels. Currently see {} labels, namely {} ".format(self.num_labels, self.labels))
        else:
            if self.multilabel:
                self.logger.info("Instantiated multi-label classification metrics for {} labels".format(self.num_labels))
            else:
                self.logger.info("Instantiated single-label classification metrics for {} labels".format(self.num_labels))

        self._create_metrics()
    
    def _create_metrics(self):
        """Instantiates the auroc, f1, precision, and recall based upon the initialization args for the class"""
        self.accuracy = Accuracy(dist_sync_on_step=True, threshold = self.classification_threshold, num_classes=self.num_labels, average="micro" if self.num_labels == 2 else "macro")
        self.precision = Precision(dist_sync_on_step=True, num_classes=self.num_labels, average="micro" if self.num_labels == 2 else "macro")
        self.recall = Recall(dist_sync_on_step=True, threshold = self.classification_threshold, num_classes=self.num_labels, average="micro" if self.num_labels == 2 else "macro")
        self.auroc = AUROC(dist_sync_on_step=True, num_classes=self.num_labels, average="macro")
        self.f1 = F1(dist_sync_on_step=True, threshold = self.classification_threshold, num_classes=self.num_labels, average="micro" if self.num_labels == 2 else "macro")
        self.metric_collection = MetricCollection([self.accuracy, self.precision, self.recall, self.auroc, self.f1])
        self.train_metrics = self.metric_collection.clone(prefix='train_')
        self.validation_metrics = self.metric_collection.clone(prefix='val_')
    
    def compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor, validation=True) -> torch.Tensor:
        #self.accuracy.update(logits.cpu(), labels.cpu())
        out = self.validation_metrics(logits.cpu(), labels.cpu()) if validation else self.train_metrics(logits.cpu(), labels.cpu())
        return out

        





    

