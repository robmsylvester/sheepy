import argparse
import os
import pickle
import pandas as pd
from typing import Any
from lib.src.common.df_ops import read_csv_text_classifier, split_dataframes, write_csv_dataset, resample_positives, resample_multilabel_positives
from lib.src.data_modules.base_data_module import BaseDataModule


class BaseCSVDataModule(BaseDataModule):
    """
    DataLoader based on the Base Classifier Data Module that represents data with pandas dataframes and an identified target column.
    """

    def __init__(self, args):
        super().__init__(args)
        self.label_encoder = None
        self.train_class_sizes = None
        # Additional columns we care about, which for now is none.
        self.args.x_cols = []
        if self.text_col is None:
            raise ValueError(
                "The base CSV data module requires a text column passed in the config file hparams with the key 'text'")

    def _verify_module(self):
        """
        Verify that the dataset shapes are correct, the class sizes and labels match the parameters, and if all is
        well, then we build the encoder
        """
        self._set_class_sizes()
        self._build_label_encoder()

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
            'label_encoder': self.label_encoder
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
        assert (self.label_encoder.vocab_size ==
                self.args.hparams["num_labels"])

    def prepare_data(self):
        """
        This is the pytorch lightning prepare_data function that is responsible for downloading data and other 
        simple operations, such as verifying its existence and reading the files. See PyTL documentation for details.
        """
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
                    fpath, evaluate=self.evaluate, label_cols=self.label_col, text_col=self.text_col,
                    additional_cols=self.args.x_cols)
                self.dataframes.append(df)
        else:
            raise ValueError(
                "Prepare_data() for the csv data module expects a data_dir of csv files")
    
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

        if isinstance(self.label_col, str): #single label
            label_dict = df[self.label_col].value_counts().to_dict()
            if positive_label is None:
                positive_label = min(label_dict, key=label_dict.get)
                self.logger.info("Inferred positive label is {}".format(positive_label))
            if resample_rate > 1:
                self.logger.info("Resampling positives...")
                return resample_positives(df,
                    resample_rate,
                    self.label_col,
                    positive_label)
            else:
                return df
        elif isinstance(self.label_col, list): #multilabel
            if isinstance(resample_rate, int):
                resample_rate = {k: resample_rate for k in self.label_col}
            
            all_ones= all(v == 1 for v in resample_rate.values())
            if all_ones:
                return df
            else:
                self.logger.info("Resampling multilabel positives...")
                return resample_multilabel_positives(df, resample_rate, positive_label)
        else:
            return df

    def setup(self, stage=None):
        """
        This is the pytorch lightning setup function that is responsible for splitting data and doing other
        operations that are split across hardware. For more information, see PyTL documentation for details.
        """
        if stage == "fit" or stage == None:
            self._train_dataset, self._val_dataset, self._test_dataset = split_dataframes(
                self.dataframes, train_ratio=self.args.hparams['train_ratio'],
                validation_ratio=self.args.hparams['validation_ratio'], test_ratio=None, shuffle=True)
            
            self._train_dataset = self._resample_positive_rows(self._train_dataset)
            
            self.logger.info("\nSplit complete. (Total) Dataset Shapes:\nTrain: {}\nValidation: {}\nTest: {}".format(
                self._train_dataset.shape, self._val_dataset.shape, self._test_dataset.shape))
        else:
            self._test_dataset = pd.concat(self.dataframes)
            self._test_dataset[self.sample_id_col] = range(
                len(self._test_dataset))
            self.logger.info("\nIn evaluation mode. Dataset Shapes:\nTest: {}".format(
                self._test_dataset.shape))

    def _write_predictions_to_disk(self, df: pd.DataFrame) -> None:
        """
        writes dataframe to the output dir
        :param df: pandas dataframe
        :return: none
        """
        write_csv_dataset(df, self.args.data_output)

    def _set_class_sizes(self):
        self.train_class_sizes = self._train_dataset[self.label_col].value_counts(
        ).to_dict()
        self.logger.info("\nLabel Count (Train):\n{}".format(
            self.train_class_sizes))

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """ Return the argument parser with necessary args for this class appended to it """
        parser.add_argument("--data_dir", type=str, required=True,
                            help="Path to the directory containing the csvs, or alternatively a single csv file")
        return parser
