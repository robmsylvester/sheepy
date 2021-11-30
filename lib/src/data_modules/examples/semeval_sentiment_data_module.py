import argparse
import pandas as pd
import os
from lib.src.data_modules.base_csv_data_module import BaseCSVDataModule
from lib.src.common.collate import single_text_collate_function, CollatedSample
from lib.src.common.df_ops import read_csv_text_classifier
from torchnlp.encoders import LabelEncoder

class SemEvalSentimentDataModule(BaseCSVDataModule):
    """
    DataLoader based on the Base Classifier Data Module that represents data with pandas dataframes and an identified target column.
    This data loader will perform multiclass classification on three sentiments.
    """

    def __init__(self, args):
        super().__init__(args)
        self.semeval_web_link = "https://www.kaggle.com/azzouza2018/semevaldatadets/download"
        self.semeval_sentiment_zip_filename = os.path.join(self.args.data_dir, "archive.zip")

    def prepare_data(self):
        """
        You can download the dataset from kaggle yourself if you need it but it is included in the repo.
        """        

        train_csv_file = os.path.join(self.args.data_dir, "semeval-2017-train.csv")
        dev_csv_file = os.path.join(self.args.data_dir, "semeval-2017-dev.csv")
        test_csv_file = os.path.join(self.args.data_dir, "semeval-2017-test.csv")

        assert os.path.exists(os.path.join(train_csv_file)), "Failed to find semeval training dataset file {} in data_dir {}".format("semeval-2017-train.csv", self.args.data_dir)
        assert os.path.exists(os.path.join(dev_csv_file)), "Failed to find semeval training dataset file {} in data_dir {}".format("semeval-2017-dev.csv", self.args.data_dir)
        assert os.path.exists(os.path.join(test_csv_file)), "Failed to find semeval training dataset file {} in data_dir {}".format("semeval-2017-test.csv", self.args.data_dir)
        
        self.dataframes = [read_csv_text_classifier(train_csv_file,
            delimiter="\t",
            evaluate=self.evaluate,
            label_cols=self.label_col,
            text_col=self.text_col)]

        self.dataframes[0]["label"] = self.dataframes[0]["label"].map({'1': 'positive', '-1': 'negative', '0': 'neutral'})
        
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
