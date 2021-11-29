import argparse
import pandas as pd
import os
from lib.src.data_modules.base_csv_data_module import BaseCSVDataModule
from lib.src.common.collate import single_text_collate_function, CollatedSample
from lib.src.common.df_ops import read_csv_text_classifier
from torchnlp.encoders import LabelEncoder

class TweetSentimentDataModule(BaseCSVDataModule):
    """
    DataLoader based on the Base Classifier Data Module that represents data with pandas dataframes and an identified target column.
    This expands upon the Base Classifier by introducing data loaders that parse the labeling format.
    """

    def __init__(self, args):
        super().__init__(args)
        self.tweet_sentiment_filename = "training.1600000.processed.noemoticon.csv"

    def prepare_data(self):
        "Verifies the data exists, and reads it from the filesystem"
        if not os.path.exists(self.args.data_dir):
            os.makedirs(self.args.data_dir)

        extracted_dataset = os.path.join(self.args.data_dir, self.tweet_sentiment_filename)
        
        if not os.path.exists(extracted_dataset):
            raise ValueError("The Sentiment140 dataset is not public, but you can access it on Kaggle. Login and download the dataset from https://www.kaggle.com/kazanova/sentiment140/download, unzip this file, and place the .csv under {}".format(self.args.data_dir))

        self.logger.info("Reading dataset...")
        cols = ['sentiment','id','date','query_string','user','text']
        self.dataframes = [read_csv_text_classifier(extracted_dataset,
            encoding='latin-1',
            evaluate=self.evaluate,
            label_cols=self.label_col,
            text_col=self.text_col, 
            names=cols)]

        self.dataframes[0]["sentiment"] = self.dataframes[0]["sentiment"].map({'4': 'positive', '0': 'negative'})
        
        #We will shuffle the dataset as well since it orders all the negatives first and positives at the end
        self.dataframes[0] = self.dataframes[0].sample(frac=1).reset_index(drop=True)

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
                                            evaluate=self.evaluate)

    def _build_label_encoder(self):
        """ Builds out custom label encoder to specify logic for which outputs will be in logits layer 
        """
        if not isinstance(self._train_dataset, pd.DataFrame):
            raise NotImplementedError(
                "Currently the default label encoder function only supports pandas dataframes")
        train_labels_list = self._train_dataset[self.label_col].unique(
        ).tolist()
        assert len(train_labels_list) == self.args.hparams["num_labels"], "Passed {} to num_labels arg but see {} unique labels in train dataset".format(
            self.args.hparams["num_labels"], len(train_labels_list))
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
            "--data_dir", type=str, required=True, help="Path to the directory containing the data")
        return parser
