# This file is deprecated and not to be used

import argparse
import pandas as pd
import os
from lib_ml_framework.src.data_modules.multi_label_base_data_module import MultiLabelBaseDataModule
from lib_ml_framework.src.common.df_ops import read_csv_text_classifier, split_dataframes


class MultiLabelCSVDataModule(MultiLabelBaseDataModule):
    """
    CSV DataModule that accepts multiple possible label columns
    """

    def __init__(self, args):
        super().__init__(args)
        # Additional columns we care about, which for now is none.
        self.args.x_cols = []
        if self.text_col is None:
            raise ValueError(
                "The base CSV data module requires a text column passed in the config file hparams with the key 'text'")

    # TODO - refactor this function
    def prepare_data(self):
        if os.path.isdir(self.args.data_dir):
            csvs = [t for t in os.listdir(
                self.args.data_dir) if t.endswith(".csv")]
            if not len(csvs):
                raise ValueError(
                    "Couldn't find any csv files in {}".format(self.args.data_dir))
            self.logger.info("\nProcessing {} csv files".format(len(csvs)))
            self.dataframes = []
            for csv in csvs:
                fpath = os.path.join(self.args.data_dir, csv)

                df = read_csv_text_classifier(
                    fpath, evaluate=self.evaluate, label_cols=self.label_col, text_col=self.text_col, additional_cols=self.args.x_cols)
                self.dataframes.append(df)
        else:
            raise ValueError(
                "Prepare_data() for the csv data module expects a data_dir of csv files")

    def setup(self, stage=None):
        if stage == "fit" or stage == None:
            self._train_dataset, self._val_dataset, self._test_dataset = split_dataframes(
                self.dataframes, train_ratio=self.args.hparams['train_ratio'], validation_ratio=self.args.hparams['validation_ratio'], test_ratio=None, shuffle=True)
            
            self._train_dataset = self._resample_positive_rows(self._train_dataset)
            
            self.logger.info("\nSplit complete. (Total) Dataset Shapes:\nTrain: {}\nValidation: {}\nTest: {}".format(
                self._train_dataset.shape, self._val_dataset.shape, self._test_dataset.shape))
        else:
            self._test_dataset = pd.concat(self.dataframes)
            self._test_dataset[self.sample_id_col] = range(
                len(self._test_dataset))
            self.logger.info("\nIn evaluation mode. Dataset Shapes:\nTest: {}".format(
                self._test_dataset.shape))

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

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """ Return the argument parser with necessary args for this class appended to it """
        parser.add_argument("--data_dir", type=str, default="~/datasets/action_items/lmi_f8_v2_3",
                            help="Path to the directory containing the csvs, or alternatively a single csv file")
        return parser
