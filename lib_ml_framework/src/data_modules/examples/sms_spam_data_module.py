import argparse
import pandas as pd
import os
import requests
import zipfile
from random import shuffle
from typing import List, Tuple
from pytorch_lightning import LightningDataModule
from lib_ml_framework.src.data_modules.base_csv_data_module import BaseCSVDataModule
from lib_ml_framework.src.common.collate import single_text_collate_function, CollatedSample
from torchnlp.encoders import LabelEncoder

class SmsSpamDataModule(BaseCSVDataModule):
    """
    DataLoader based on the Base Classifier Data Module that represents data with pandas dataframes and an identified target column.
    This expands upon the Base Classifier by introducing data loaders that parse the SMS Spam Dataset
    """

    def __init__(self, args):
        super().__init__(args)
    
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

    def prepare_data(self):
        "Downloads the data if it doesn't exist. Unzips it, and deletes the zip file. Reads the data file"
        if not os.path.exists(self.args.data_dir):
            os.makedirs(self.args.data_dir)
        
        if not os.path.exists(os.path.join(self.args.data_dir, "SMSSpamCollection")):
            #Download
            self.logger.info("Downloading SMS Spam Dataset")

            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
            r = requests.get(url, allow_redirects=True)

            #Unzip
            self.logger.info("Unzipping SMS Spam Dataset")
            zip_dataset = "smsspamcollection.zip"
            open(zip_dataset, 'wb').write(r.content)
            with zipfile.ZipFile(zip_dataset, 'r') as zip_ref:
                zip_ref.extractall(self.args.data_dir)
            
            #Clean
            os.remove(zip_dataset)
            self.logger.info("Finished extracting dataset")
        
        assert os.path.exists(os.path.join(self.args.data_dir, "SMSSpamCollection")), "Failed to find SMSSpamCollection in data_dir {}".format(self.args.data_dir)

        self.logger.info("Reading dataset...")
        self.dataframes = [self._read_dataset()]

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
        return single_text_collate_function(sample,
                                            self.text_col,
                                            self.label_col,
                                            self.sample_id_col,
                                            self.nlp['tokenizer'],
                                            self.label_encoder,
                                            prepare_target=self.prepare_target,
                                            prepare_sample_id=self.evaluate)

    def _build_label_encoder(self):
        """ Builds out custom label encoder to specify logic for which outputs will be in logits layer """
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
            "--data_dir", type=str, required=True, help="Path to the directory containing the dataset as separate files.")
        return parser
