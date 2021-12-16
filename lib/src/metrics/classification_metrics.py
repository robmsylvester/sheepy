import torch
from typing import List, Any
from lib.src.common.logger import get_std_out_logger
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, AUROC, F1, ROC, AUC, AveragePrecision

#TODO - need to add LabelEncoder/dict on here if user passes pos_label argument in metrics so we can decode it to get its integer representation
class ClassificationMetrics():
    def __init__(self,
        labels: List[Any],
        config: dict,
        logger=None):
        """A class to help abstract the classification metrics that are reported and sent to PyTorch Lightning's
        validation epoch callbacks as well as Weights and Biases visualizations.

        Args:
            labels (List[Any]): List of labels in the dataset you expect to see. Needed to instantiate metric classes
            config (Dict): Metric Config JSON from your experiment config profile
            logger ([Any], optional): Logger to console. Defaults to None to get standard logger
        """
        self.num_labels = len(labels)
        self.logger = get_std_out_logger() if logger is None else logger
        self.config = config

        self.torchmetrics_map = {
            'auroc': AUROC,
            'f1': F1,
            'precision': Precision,
            'recall': Recall,
            'accuracy': Accuracy,
            'roc': ROC,
            'average_precision': AveragePrecision,
            'auc': AUC,
        }

        self._create_metrics()
    
    def _create_metrics(self):
        """Instantiates the auroc, f1, precision, and recall based upon the initialization args for the class"""
        tracked_metrics = []
        for metric_dict in self.config:
            if metric_dict['name'] in self.torchmetrics_map:
                metric_class = self.torchmetrics_map[metric_dict['name']]
                args_dict = metric_dict.copy()
                del args_dict['name']
                args_dict['num_classes'] = self.num_labels
                args_dict['dist_sync_on_step']=True
                metric = metric_class(**args_dict)
                tracked_metrics.append(metric)
            else:
                self.logger.warn("Metric {} not found and will be ignored. See classification_metrics.py and verify that your metric is mapped to a torchmetrics class")

        self.metric_collection = MetricCollection(tracked_metrics)
        self.train_metrics = self.metric_collection.clone(prefix='train/')
        self.validation_metrics = self.metric_collection.clone(prefix='val/')
        self.test_metrics = self.metric_collection.clone(prefix='test/')
    
    def compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor, stage: str) -> torch.Tensor:
        if stage == "train" or stage == "train_epoch":
            out = self.train_metrics(logits.cpu(), labels.cpu())
        elif stage == "val" or stage == "val_epoch":
            out = self.validation_metrics(logits.cpu(), labels.cpu())
        elif stage == "test" or stage == "test_epoch":
            out = self.test_metrics(logits.cpu(), labels.cpu())
        else:
            raise ValueError("stage to compute metrics must be either 'train', 'val', or 'test', 'train_epoch', 'val_epoch', or 'test_epoch'. Instead, passed {}".format(stage))
        return out

        





    

