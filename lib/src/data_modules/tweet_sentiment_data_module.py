import argparse
import pandas as pd
import os
from typing import List
from pytorch_lightning import LightningDataModule
from lib.src.data_modules.base_csv_data_module import BaseCSVDataModule
from lib.src.common.collate import single_text_collate_function, CollatedSample
from torchnlp.encoders import LabelEncoder

class TweetSentimentDataModule(BaseCSVDataModule):
    """
    DataLoader based on the Base Classifier Data Module that represents data with pandas dataframes and an identified target column.
    This expands upon the Base Classifier by introducing data loaders that parse the labeling format.
    """

    def __init__(self, args):
        super().__init__(args)

    def prepare_sample(self, sample: list) -> CollatedSample:
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
                                            self.label_encoder,
                                            prepare_target=self.prepare_target,
                                            prepare_sample_id=self.evaluate)

    def _build_label_encoder(self):
        """ Builds out custom label encoder to specify logic for which outputs will be in logits layer 
        """
        if not isinstance(self._train_dataset, pd.DataFrame):
            raise NotImplementedError(
                "Currently the default label encoder function only supports pandas dataframes")
        train_labels_list = self._train_dataset[self.label_col].unique(
        ).tolist()
        assert len(train_labels_list) == self.args.hparams["num_labels"], "Passed {} to num_labels arg but see {} unique labels in train dataset".format(
            self.args.num_labels, len(train_labels_list))
        self.label_encoder = LabelEncoder(
            train_labels_list,
            reserved_labels=[])
        self.label_encoder.unknown_index = 0
        self.logger.info("\nEncoded Labels:\n{}".format(
            self.label_encoder.vocab))
        assert self.label_encoder.vocab_size == self.args.hparams["num_labels"]

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """ Return the argument parser with necessary args for this class appended to it """
        parser.add_argument(
            "--data_dir", type=str, required=True, help="Path to the directory containing the synthetic transcripts as separate files.")
        return parser
