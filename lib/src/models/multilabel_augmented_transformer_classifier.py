import argparse
from pytorch_lightning import LightningDataModule
from lib.src.models.multilabel_transformer_classifier import MultiLabelTransformerClassifier
from lib.src.models.augmented_transformer_classifier import AugmentedTransformerClassifier

class MultiLabelAugmentedTransformerClassifier(AugmentedTransformerClassifier, MultiLabelTransformerClassifier):
    """
    The multilabel rich transformer classifier behaves (overrides) correctly under multiple inheritance if the parent
    classes are ordered in the way above. 
    """
    def __init__(self, args: argparse.Namespace, data: LightningDataModule) -> None:
        super().__init__(args, data)

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """ Return the argument parser with necessary args for this class appended to it """
        return parser
