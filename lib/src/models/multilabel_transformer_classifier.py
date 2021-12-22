import argparse
from pytorch_lightning import LightningDataModule
from lib.src.models.multilabel_base_classifier import MultiLabelBaseClassifier
from lib.src.models.base_transformer_classifier import TransformerClassifier

class MultiLabelTransformerClassifier(MultiLabelBaseClassifier, TransformerClassifier):
    def __init__(self, args: argparse.Namespace, data: LightningDataModule) -> None:
        super().__init__(args, data)
    
    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """ Return the argument parser with necessary args for this class appended to it """
        return parser