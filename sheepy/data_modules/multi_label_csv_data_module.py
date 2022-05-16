import argparse
import os
import pickle
from typing import Any, List

import pandas as pd

from sheepy.common.df_ops import resample_multilabel_positives
from sheepy.data_modules.base_csv_data_module import BaseCSVDataModule
from sheepy.nlp.label_encoder import LabelEncoder


class MultiLabelCSVDataModule(BaseCSVDataModule):
    """
    CSV DataModule that accepts multiple possible label columns
    """

    def __init__(self, args):
        super().__init__(args)
        # Additional columns we care about, which for now is none.
        if self.text_col is None:
            raise ValueError(
                "The base CSV data module requires a text column passed in the config file hparams with the key 'text'")
        self._verify_multilabel()

    def _verify_multilabel(self):
        assert isinstance(
            self.args.hparams["label"], list), "hyperparameter of labels must be a list for the multi label module to be used"
        assert len(
            self.args.hparams["label"]) > 1, "there must be more than one label in the list for the multi label module to be used"
        assert self.args.hparams["num_labels"] == len(self.args.hparams["label"]), "config sees {} labels but num_labels set to {}".format(
            len(self.args.hparams["label"]), self.args.hparams["num_labels"])

    def _resample_positive_rows(self, df: pd.DataFrame, positive_label: Any="1") -> pd.DataFrame:
        """Calls the df_ops resample function for a given dataframe, subject to the parameter arguments
        provided to the experiment. Assumes that the sparse label is the positive label. If this is not
        the behavior you want, then you'll need to modify this. Ignores calling the function if
        positive_resample_rate is not set or is set to 1.

        Unlike the binary _resample_positive_rows for a single label, the multilabel version reads
        a dictionary in the config  with a resample rate for each label. If an
        integer is provided instead, this integer will be used for each of the labels.

        If the positive label is None, it will be inferred as the sparse label

        Args:
            df (pd.DataFrame): the training dataset, or some dataframe to sample.
            positive_label (int): the identifier for the positive label. #TODO - this should probably be a dict too?

        Returns:
            pd.DataFrame: Dataframe with resamples rows added
        """
        resample_rate = self.args.hparams.get('positive_resample_rate', 1)
        if isinstance(resample_rate, dict):
            self.logger.info("Resampling multilabel positives...")
            return resample_multilabel_positives(df, resample_rate, positive_label)
        elif isinstance(resample_rate, int):
            resample_rate = {k: resample_rate for k in self.label_col}

            all_ones= all(v == 1 for v in resample_rate.values())
            if all_ones:
                return df
            else:
                self.logger.info("Resampling multilabel positives...")
                return resample_multilabel_positives(df, resample_rate, positive_label)
        else:
            return df

    def _maybe_map_labels(self):
        pass

    def _load_text_from_text_file(self, filepath: str) -> List[dict]:
        """[summary]

        Args:
            filepath (str): [description]

        Returns:
            List[dict]: [description]
        """
        prepared_inputs = []
        with open(filepath, "r") as f:
            for idx, line in enumerate(f):
                line = line.rstrip()
                prepared_input = {k: None for k in self.label_col}
                prepared_input[self.text_col] = line
                prepared_input[self.sample_id_col] = idx
                prepared_inputs.append(prepared_input)
        return prepared_inputs

    def _load_text_from_raw_input(self) -> List[dict]:
        sample_text = input("Enter sample text. (Press Ctrl+C to exit)\n")
        prepared_input = {self.text_col: sample_text, self.sample_id_col: 0}
        for col in self.label_col:
            prepared_input[col] = None
        return [prepared_input]

    def save_data_module(self, out_path: str = None):
        """
        Pickles the module to a specified output path.

        Args:
            out_path - str, where to save the data module. You should probably leave this as None and
            rely on default behavior.
        """
        if out_path is None:
            out_path = os.path.join(self.args.output_dir, "data.module")
        module_dict = {
            'class_sizes': self.train_class_sizes,
            'label_encoder': self.label_encoder,
            'pos_weights': self.pos_weights,
        }
        with open(out_path, 'wb') as fp:
            pickle.dump(module_dict, fp)

    def load_data_module(self, in_path: str = None):
        if in_path is None:
            in_path = os.path.join(self.args.output_dir, "data.module")
        with open(in_path, 'rb') as fp:
            module_dict = pickle.load(fp)
            self.train_class_sizes = module_dict['class_sizes']
            self.label_encoder = module_dict['label_encoder']
            self.pos_weights = module_dict['pos_weights']
        assert (self.label_encoder.vocab_size ==
                self.args.hparams["num_labels"])

    #TODO - this one is probably good
    def _set_class_sizes(self, positive_label="1", negative_label="0"):
        """
        Searches through the labels in the dataset and counts the number of positives/negatives for each.
        This helps set the weights for the loss function, and also establishes a nice sanity
        check for the user on label counts.
        """
        self.train_class_sizes = {}
        self.pos_weights = {}
        for col in self.label_col:
            if not self._train_dataset[col].nunique() == 2:
                raise ValueError("Saw {} unique values in label {}. Each individual label should be binary".format(
                    self._train_dataset[col].nunique(), col))
            pos_size = self._train_dataset[self._train_dataset[col]
                                           == positive_label].shape[0]
            neg_size = self._train_dataset[self._train_dataset[col]
                                           == negative_label].shape[0]
            self.train_class_sizes[col] = pos_size
            self.pos_weights[col] = float(neg_size) / pos_size
        self.logger.info("\nPositive Label Count (Train):\n{}".format(
            self.train_class_sizes))
        self.logger.info("\nPositive Label Weights (Train): (Ratio of Neg/Pos)\n{}".format(
            self.pos_weights))

    def _build_label_encoder(self):
        self.label_encoder = LabelEncoder.initializeFromMultilabelDataframe(self._train_dataset, self.args.hparams["label"])
        self.logger.info("Label Encoder Vocab: {}".format(self.label_encoder.vocab))

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """ Return the argument parser with necessary args for this class appended to it """
        parser.add_argument("--data_dir", type=str, required=False,
                            help="If all of your data is in one folder, this is the path to that directory containing the csvs, or alternatively a single csv file")
        parser.add_argument("--train_data_dir", type=str, required=False,
                            help="If your data is split across folders, this is the path to the directory containing the training csvs, or alternatively a single train csv file")
        parser.add_argument("--val_data_dir", type=str, required=False,
                            help="If your data is split across folders, this is the path to the directory containing the validation csvs, or alternatively a single validation csv file")
        parser.add_argument("--test_data_dir", type=str, required=False,
                            help="If your data is split across folders, this is the path to the directory containing the test csvs, or alternatively a single test csv file")

        return parser
