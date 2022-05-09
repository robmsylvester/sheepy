import argparse
from typing import List

from sheepy.src.data_modules.base_data_module import BaseDataModule


class MultiLabelBaseDataModule(BaseDataModule):
    """
    Multi-class Multi-Label Data Module. This class is a drop in replacement for the
    base classification data module, so if you want to write multilabel data modules
    for csv's, json's, and other dataset types, they can inherit directly from this class

    Does not have explicit logits for negative label. In other words, each logit specifies
    a unique class encountered for each label, with a restriction that each of the classes
    is a binary label. If this is not the case, you'll have to either create binary dummies
    for these variables or write your own data module.

    Each logit has a training loss that scales the weight of the positive by the ratio of
    negatives to positives for that label.

    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.args = args
        self._verify_multilabel()

    def _verify_multilabel(self):
        assert isinstance(
            self.args.hparams["label"], list), "hyperparameter of labels must be a list for the multi label module to be used"
        assert len(
            self.args.hparams["label"]) > 1, "there must be more than one label in the list for the multi label module to be used"
        assert self.args.hparams["num_labels"] == len(self.args.hparams["label"]), "config sees {} labels but num_labels set to {}".format(
            len(self.args.hparams["label"]), self.args.hparams["num_labels"])


    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """ Return the argument parser with necessary args for this class appended to it """
        parser.add_argument("--disable_weight_scale", action="store_true",
                            help="Disables automatically scaling class weights from training set")
        return parser
