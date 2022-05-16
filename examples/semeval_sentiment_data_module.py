import argparse
import os
from typing import List

import pandas as pd

from sheepy.common.collate import CollatedSample, single_text_collate_function
from sheepy.common.df_ops import read_csv_text_classifier
from sheepy.common.s3_ops import download_resource
from sheepy.data_modules.base_csv_data_module import BaseCSVDataModule


class SemEvalSentimentDataModule(BaseCSVDataModule):
    """
    DataLoader based on the Base Classifier Data Module that represents data with pandas dataframes and an identified target column.
    This data loader will perform multiclass classification on three sentiments.
    """

    def __init__(self, args):
        super().__init__(args)
        self.all_dataframes, self.train_dataframes, self.val_dataframes, self.test_dataframes = None, None, None, None

    def prepare_data(self, stage: str=None):
        """[summary]

        Args:
            stage (str, optional): [description]. Defaults to None.
        """

        if self.evaluate:
            return

        if not os.path.isfile(self.args.train_data_dir) or not self.args.train_data_dir.endswith(".csv"):
            self.logger.info("Downloading semeval train data")
            KEY = 'datasets/semeval/semeval-2017-train.csv'
            _ = download_resource(KEY, self.args.train_data_dir)

        if not os.path.isfile(self.args.val_data_dir) or not self.args.val_data_dir.endswith(".csv"):
            self.logger.info("Downloading semeval val data")
            KEY = 'datasets/semeval/semeval-2017-val.csv'
            _ = download_resource(KEY, self.args.val_data_dir)

        if not os.path.isfile(self.args.test_data_dir) or not self.args.test_data_dir.endswith(".csv"):
            self.logger.info("Downloading semeval test data")
            KEY = 'datasets/semeval/semeval-2017-test.csv'
            _ = download_resource(KEY, self.args.test_data_dir)

    def setup(self, stage: str=None):
        """[summary]

        Args:
            stage (str, optional): [description]. Defaults to None.
        """
        # self.all_dataframes, self.train_dataframes, self.val_dataframes, self.test_dataframes = None, None, None, None
        # if hasattr(self.args, 'data_dir'): #A single directory is passed with all the data
        #     self.all_dataframes = self._read_csv_directory(self.args.data_dir)
        self.train_dataframes = self._read_csv_directory(self.args.train_data_dir)
        self.val_dataframes = self._read_csv_directory(self.args.val_data_dir)
        self.test_dataframes = self._read_csv_directory(self.args.test_data_dir)
        self._train_dataset, self._val_dataset, self._test_dataset = self.split_dataframes(train_dataframes=self.train_dataframes, val_dataframes=self.val_dataframes, test_dataframes=self.test_dataframes, stage=stage)


    def _read_csv_directory(self, csv_path: str) -> List[pd.DataFrame]:
        """Given a directory of csv files, or a single csv file, return a list of pandas dataframes containing
        the data according to the config profile of the experiment specifying the targeted columns.

        Args:
            csv_path (str): folder containing the .csv files, or a single .csv file

        Returns:
            List[pd.DataFrame]: List of pandas dataframes containing necessary data in each csv
        """
        if os.path.isdir(csv_path):
            csvs = [t for t in os.listdir(csv_path) if t.endswith(".csv")]
        elif csv_path.endswith(".csv"):
            csvs = [csv_path]
        else:
            raise ValueError("csv path {} is neither a csv file nor a directory".format(csv_path))
        if not len(csvs):
            raise ValueError("Couldn't find any csv files in {}".format(csv_path))
        out_dfs = []
        for csv_file in csvs:
            fpath = csv_file if csv_file.endswith(".csv") else os.path.join(csv_path, csv_file)
            df = read_csv_text_classifier(
                fpath, evaluate=self.evaluate, delimiter="\t", label_cols=self.label_col, text_col=self.text_col)
            out_dfs.append(df)
        return out_dfs

    def prepare_sample(self, sample: list) -> CollatedSample:
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.

        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        if self.text_col is None:
            raise NotImplementedError(
                "To use the default collate function you need a text column in your hparams config under the key 'text', or some other way of preparing your sample from other functions.")
        return single_text_collate_function(sample,
                                            self.text_col,
                                            self.label_col,
                                            self.sample_id_col,
                                            self.tokenizer,
                                            self.label_encoder,
                                            evaluate=self.evaluate)

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """ Return the argument parser with necessary args for this class appended to it """
        parser.add_argument("--train_data_dir", type=str, required=False,
                            help="If your data is split across folders, this is the path to the directory containing the training csvs, or alternatively a single train csv file")
        parser.add_argument("--val_data_dir", type=str, required=False,
                            help="If your data is split across folders, this is the path to the directory containing the validation csvs, or alternatively a single validation csv file")
        parser.add_argument("--test_data_dir", type=str, required=False,
                            help="If your data is split across folders, this is the path to the directory containing the test csvs, or alternatively a single test csv file")
        return parser
