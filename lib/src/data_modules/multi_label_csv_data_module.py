import argparse
import pandas as pd
from typing import Any
from lib.src.data_modules.base_csv_data_module import BaseCSVDataModule
from lib.src.nlp.label_encoder import LabelEncoder
from lib.src.common.df_ops import read_csv_text_classifier, split_dataframes, resample_multilabel_positives


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
    
    def _maybe_map_labels(self):
        pass

    # # TODO - REPLACE
    # def prepare_data(self):
    #     if os.path.isdir(self.args.data_dir):
    #         csvs = [t for t in os.listdir(
    #             self.args.data_dir) if t.endswith(".csv")]
    #         if not len(csvs):
    #             raise ValueError(
    #                 "Couldn't find any csv files in {}".format(self.args.data_dir))
    #         self.logger.info("\nProcessing {} csv files".format(len(csvs)))
    #         self.dataframes = []
    #         for csv in csvs:
    #             fpath = os.path.join(self.args.data_dir, csv)

    #             df = read_csv_text_classifier(
    #                 fpath, evaluate=self.evaluate, label_cols=self.label_col, text_col=self.text_col, additional_cols=self.args.x_cols)
    #             self.dataframes.append(df)
    #     else:
    #         raise ValueError(
    #             "Prepare_data() for the csv data module expects a data_dir of csv files")

    # #TODO - REPLACE
    # def setup(self, stage=None):
    #     if stage == "fit" or stage == None:
    #         self._train_dataset, self._val_dataset, self._test_dataset = split_dataframes(
    #             self.dataframes, train_ratio=self.args.hparams['train_ratio'], validation_ratio=self.args.hparams['validation_ratio'], test_ratio=None, shuffle=True)
            
    #         self._train_dataset = self._resample_positive_rows(self._train_dataset)
            
    #         self.logger.info("\nSplit complete. (Total) Dataset Shapes:\nTrain: {}\nValidation: {}\nTest: {}".format(
    #             self._train_dataset.shape, self._val_dataset.shape, self._test_dataset.shape))
    #     else:
    #         self._test_dataset = pd.concat(self.dataframes)
    #         self._test_dataset[self.sample_id_col] = range(
    #             len(self._test_dataset))
    #         self.logger.info("\nIn evaluation mode. Dataset Shapes:\nTest: {}".format(
    #             self._test_dataset.shape))

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
