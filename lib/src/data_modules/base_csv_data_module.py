import argparse
import os
import pickle
import pandas as pd
from typing import Any, List
from lib.src.common.df_ops import read_csv_text_classifier, split_dataframes, write_csv_dataset, resample_positives, resample_multilabel_positives, map_labels
from lib.src.data_modules.base_data_module import BaseDataModule


class BaseCSVDataModule(BaseDataModule):
    """
    DataLoader based on the Base Classifier Data Module that represents data with pandas dataframes and an identified target column.
    """

    def __init__(self, args):
        super().__init__(args)
        self.label_encoder = None
        self.train_class_sizes = None
        self._validate_data_dir_args()
        self.args.x_cols = [] # Additional columns we care about, which for now is none.

        if self.text_col is None:
            raise ValueError(
                "The base CSV data module requires a text column passed in the config file hparams with the key 'text'")

    def _validate_data_dir_args(self):
        """Validates that either a data_dir arg has been passed, or alternatively separate arguments have been passed
        specifying the location of the data directories for the train, validation, and test csv's.

        Raises:
            ValueError- If arguments are an invalid combination
        """
        if self.evaluate:
            return
        if hasattr(self.args, 'data_dir'):
            if not os.path.exists(self.args.data_dir):
                raise ValueError("The passed directory for data_dir does not exist: {}".format(self.args.data_dir))
            self.logger.info("Using single filesystem data location {}".format(self.args.data_dir))
        else:
            if not hasattr(self.args, 'train_data_dir') or not os.path.exists(self.args.train_data_dir):
                raise ValueError("The passed directory for train_data_dir does not exist: {}".format(self.args.train_data_dir))
            if not hasattr(self.args, 'val_data_dir') or not os.path.exists(self.args.val_data_dir):
                raise ValueError("The passed directory for val_data_dir does not exist: {}".format(self.args.val_data_dir))
            if not hasattr(self.args, 'test_data_dir') or not os.path.exists(self.args.test_data_dir):
                raise ValueError("The passed directory for test_data_dir does not exist: {}".format(self.args.test_data_dir))
            self.logger.info("Using separate filesystem data locations for train/val/test.\n\ttrain:{}\n\tval:{}\n\ttest:{}".format(self.args.train_data_dir, self.args.val_data_dir, self.args.test_data_dir))

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
                fpath, evaluate=self.evaluate, label_cols=self.label_col, text_col=self.text_col,
                additional_cols=self.args.x_cols)
            out_dfs.append(df)
        return out_dfs

    def prepare_data(self):
        """
        This is the pytorch lightning prepare_data function that is responsible for downloading data and other 
        simple operations, such as verifying its existence and reading the files. See PyTL documentation for details.
        """

        self.all_dataframes, self.train_dataframes, self.val_dataframes, self.test_dataframes = None, None, None, None
        if hasattr(self.args, 'data_dir'): #A single directory is passed with all the data
            self.all_dataframes = self._read_csv_directory(self.args.data_dir)
        else: #separate directories are passed for train/val/test data
            self.train_dataframes = self._read_csv_directory(self.args.train_data_dir)
            self.val_dataframes = self._read_csv_directory(self.args.val_data_dir)
            self.test_dataframes = self._read_csv_directory(self.args.test_data_dir)
    
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

    def setup(self, stage: str=None):
        """This is the pytorch lightning setup function that is responsible for splitting data and doing other
        operations that are split across hardware. For more information, see PyTL documentation for details.

        This specific function will 

        Args:
            stage ([str], optional): The PyTL stage. Defaults to None.
        """
        if stage == "fit" or stage == None:
            if self.all_dataframes is None:
                self._train_dataset, _, _ = split_dataframes(self.train_dataframes, train_ratio=1., validation_ratio=0., test_ratio=0., shuffle=True)
                _, self._val_dataset, _ = split_dataframes(self.val_dataframes, train_ratio=0., validation_ratio=1., test_ratio=0., shuffle=True)
                _, _, self._test_dataset = split_dataframes(self.test_dataframes, train_ratio=0., validation_ratio=0., test_ratio=1., shuffle=True)
            else:
                self._train_dataset, self._val_dataset, self._test_dataset = split_dataframes(
                    self.all_dataframes, train_ratio=self.args.hparams['train_ratio'],
                    validation_ratio=self.args.hparams['validation_ratio'], test_ratio=None, shuffle=True)
                
            self._train_dataset = self._resample_positive_rows(self._train_dataset)

            if 'label_map' in self.args.hparams and isinstance(self.args.hparams['label_map'], dict):
                self.logger.info("Using label map {}".format(self.args.hparams['label_map']))
                self._train_dataset = map_labels(self._train_dataset, self.label_col, self.args.hparams['label_map'])
                self._val_dataset = map_labels(self._val_dataset, self.label_col, self.args.hparams['label_map'])
                self._test_dataset = map_labels(self._test_dataset, self.label_col, self.args.hparams['label_map'])
            
            self.logger.info("Dataset split complete. (Total) Dataset Shapes:\n\tTrain: {}\nV\talidation: {}\n\tTest: {}".format(
                self._train_dataset.shape, self._val_dataset.shape, self._test_dataset.shape))
        else: #evaluating
            self._test_dataset = pd.concat(self.all_dataframes if self.all_dataframes is not None else self.test_dataframes)
            self._test_dataset[self.sample_id_col] = range(
                len(self._test_dataset))
            self.logger.info("In evaluation mode. Dataset Shapes:\n\tTest: {}".format(
                self._test_dataset.shape))

    def _set_class_sizes(self):
        self.train_class_sizes = self._train_dataset[self.label_col].value_counts(
        ).to_dict()
        self.logger.info("\nLabel Count (Train):\n{}".format(
            self.train_class_sizes))

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
