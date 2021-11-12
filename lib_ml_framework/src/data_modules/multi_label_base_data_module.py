import argparse
import pandas as pd
from typing import List
from lib_ml_framework.src.data_modules.base_data_module import BaseDataModule
from lib_ml_framework.src.common.collate import CollatedSample, single_text_collate_function


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

    def _build_label_encoder(self):
        """
        Builds out custom label encoder to specify logic for which outputs will be in logits layer.
        Because this is multilabel, our label encoder is just a list of the binary columns. There
        really isn't any encoding/decoding we need. If we remove the binary restriction, we'll need one.

        For now, this method just implements some data verification logic that probably is better suited
        for prepare_data() anyway.
        """
        # TODO - move this to prepare_data in a sane way
        if not isinstance(self._train_dataset, pd.DataFrame):
            raise NotImplementedError(
                "Currently the default label encoder function only supports pandas dataframes")
        assert len(self.args.hparams["label"]
                   ) == self.args.hparams["num_labels"]
        for label in self.args.hparams["label"]:
            unique_vals = self._train_dataset[label].unique()
            if len(unique_vals) > 2:  # this restriction can probably be removed eventually
                raise ValueError("Label {} must be binary. See values {}".format(
                    label, str(unique_vals)))

    def prepare_sample(self, sample: List) -> CollatedSample:
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.

        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        if not hasattr(self, "nlp"):
            raise ValueError(
                "Missing attribute nlp on data module object. It is likely the nlp() method has not been called.")
        if self.text_col is None:
            raise NotImplementedError(
                "To use the default collate function you need a text column in your hparams config under the key 'text', or some other way of preparing your sample from other functions.")
        return single_text_collate_function(sample,
                                            self.text_col,
                                            self.label_col,
                                            self.sample_id_col,
                                            self.nlp['tokenizer'],
                                            prepare_target=self.prepare_target,
                                            prepare_sample_id=self.evaluate)

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """ Return the argument parser with necessary args for this class appended to it """
        parser.add_argument("--disable_weight_scale", action="store_true",
                            help="Disables automatically scaling class weights from training set")
        return parser
