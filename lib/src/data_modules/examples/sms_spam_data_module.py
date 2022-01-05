import argparse
import pandas as pd
import os
from lib.src.data_modules.base_csv_data_module import BaseCSVDataModule
from lib.src.common.collate import single_text_collate_function, CollatedSample
from lib.src.common.s3_ops import download_resource


class SmsSpamDataModule(BaseCSVDataModule):
    """
    DataLoader based on the Base Classifier Data Module that represents data with pandas dataframes and an identified target column.
    This expands upon the Base Classifier by introducing data loaders that parse the SMS Spam Dataset
    """

    def __init__(self, args):
        super().__init__(args)
        self.all_dataframes, self.train_dataframes, self.val_dataframes, self.test_dataframes = None, None, None, None
    
    def _read_dataset(self) -> pd.DataFrame:
        """Reads the dataset file that contains all the SMS spam, and returns a list of tuples

        Returns:
            pd.DataFrame: Dataframe of examples with text and label columns
        """
        
        column_names = [self.text_col, self.label_col]
        dataset = pd.DataFrame(columns=column_names)
        with open(os.path.join(self.args.data_dir, "SMSSpamCollection"), "r") as data_file:
            for line in data_file:
                line = line.strip().split("\t")
                assert len(line)==2
                label, text = line[0], line[1]
                dataset = dataset.append({self.text_col: text, self.label_col: label}, ignore_index=True)

        return dataset
    
    def _read_evaluation_file(self, filepath: str) -> pd.DataFrame:
        """Reads a file that contains text only on each line

        Returns:
            pd.DataFrame: Dataframe of examples with text column
        """
        column_names = [self.text_col]
        dataset = pd.DataFrame(columns=column_names)
        with open(filepath, "r") as data_file:
            for line in data_file:
                text = line.strip()
                dataset = dataset.append({self.text_col: text}, ignore_index=True)
        return dataset

    def prepare_data(self, stage: str=None):
        """[summary]

        Args:
            stage (str, optional): [description]. Defaults to None.
        """
        if self.evaluate:
            return
        
        if not os.path.exists(self.args.data_dir):
            os.makedirs(self.args.data_dir)
        
        if not os.path.exists(os.path.join(self.args.data_dir, "SMSSpamCollection")):
            self.logger.info("Downloading SMS Spam Dataset")
            KEY = 'datasets/spam/SMSSpamCollection'
            dst = os.path.join(self.args.data_dir, "SMSSpamCollection")
            _ = download_resource(KEY, dst)

        assert os.path.exists(os.path.join(self.args.data_dir, "SMSSpamCollection")), "Failed to find SMSSpamCollection in data_dir {}".format(self.args.data_dir)

    def setup(self, stage: str=None):
        """[summary]

        Args:
            stage (str, optional): [description]. Defaults to None.
        """
        if self.evaluate:
            self.logger.info("Reading evaluation dataset directory {}".format(self.args.data_dir))
            self.all_dataframes = []
            for f in os.listdir(self.args.data_dir):
                if f.endswith(".txt"):
                    self.all_dataframes.append(self._read_evaluation_file(os.path.join(self.args.data_dir, f)))
        else:
            self.logger.info("Reading dataset...")
            self.all_dataframes = [self._read_dataset()]
        
        self._train_dataset, self._val_dataset, self._test_dataset = self.split_dataframes(all_dataframes=self.all_dataframes, stage=stage)

    def prepare_sample(self, sample: list) -> CollatedSample:
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.

        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
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
        parser.add_argument("--data_dir", type=str, required=True,
                            help="If all of your data is in one folder, this is the path to that directory containing the csvs, or alternatively a single csv file")
        return parser
