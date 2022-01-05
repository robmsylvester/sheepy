import os
import argparse
import pandas as pd
from lib.src.common.s3_ops import download_resource
from lib.src.common.df_ops import split_dataframes
from lib.src.data_modules.multi_label_csv_data_module import MultiLabelCSVDataModule

class ToxicCommentDataModule(MultiLabelCSVDataModule):
    """
    DataLoader based on the multilabel csv data module provoiding an example of multi label ETL
    """
    def __init__(self, args):
        super().__init__(args)
        self._train_dataset, self._val_dataset, self._test_dataset = None, None, None

    def _read_dataset(self, data_dir: str) -> pd.DataFrame:
        """[summary]ValueError: The implied number of classes (from shape of inputs) does not match num_classes.

        Args:
            data_dir (str): [description]

        Returns:
            pd.DataFrame: [description]
        """
        pass
    
    def prepare_data(self, stage=None):
        """[summary]

        Args:
            stage ([type], optional): [description]. Defaults to None.
        """
        train_dst = os.path.join(self.args.data_dir, "train.csv")
        test_dst = os.path.join(self.args.data_dir, "test_filtered.csv")
        train_sample_dst = os.path.join(self.args.data_dir, "train_sample.csv")
        test_sample_dst = os.path.join(self.args.data_dir, "test_filtered_sample.csv")

        if self.evaluate:
            return
        
        if not os.path.exists(self.args.data_dir):
            os.makedirs(self.args.data_dir)
        
        if not os.path.exists(train_dst):
            self.logger.info("Downloading train.csv")
            KEY = 'datasets/toxic_comments/train.csv'
            _ = download_resource(KEY, train_dst)

        if not os.path.exists(test_dst):
            self.logger.info("Downloading test_filtered.csv")
            KEY = 'datasets/toxic_comments/test_filtered.csv'
            _ = download_resource(KEY, test_dst)
        
        if not os.path.exists(train_dst):
            self.logger.info("Downloading train_sample.csv")
            KEY = 'datasets/toxic_comments/train_sample.csv'
            _ = download_resource(KEY, train_sample_dst)

        if not os.path.exists(test_dst):
            self.logger.info("Downloading test_filtered_sample.csv")
            KEY = 'datasets/toxic_comments/test_filtered_sample.csv'
            _ = download_resource(KEY, test_sample_dst)
        

    def setup(self, stage=None):
        """[summary]

        Args:
            stage ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        self.all_dataframes = None
        if stage == "fit" or stage == None:
            self.train_dataframes = self._read_csv_directory(os.path.join(self.args.data_dir, "train.csv"))
            self._train_dataset, self._val_dataset, _ = split_dataframes(self.train_dataframes, train_ratio=self.args.hparams['train_ratio'], validation_ratio=self.args.hparams['validation_ratio'], test_ratio=0., shuffle=True)

            #The test set is a bit funky, but we process it separately and remove the unlabeled examples. that's why test_filtered.csv exists
            self.test_dataframes = self._read_csv_directory(os.path.join(self.args.data_dir, "test_filtered.csv"))
            _, _, self._test_dataset = split_dataframes(self.test_dataframes, train_ratio=0., validation_ratio=0., test_ratio=1., shuffle=True)

            self._train_dataset = self._resample_positive_rows(self._train_dataset)

            self.logger.info("Dataset split complete. (Total) Dataset Shapes:\n\tTrain: {}\n\tValidation: {}\n\tTest: {}".format(
                self._train_dataset.shape, self._val_dataset.shape, self._test_dataset.shape))


            train_label_size_out = "Training Dataset Sample Size For Each Label:\n"
            for label in self.args.hparams["label"]:
                support = self._train_dataset[self._train_dataset[label]=="1"].shape[0]
                train_label_size_out += "\t{}: {} samples ({} %)\n".format(label, support, round(100*support/self._train_dataset.shape[0], 4))
            self.logger.info(train_label_size_out)
            
            self._test_dataset[self.sample_id_col] = range(
                len(self._test_dataset))

        else: #evaluating
            if self._test_dataset is None:
                for f in os.listdir(self.args.data_dir):
                    if f.endswith(".txt"):
                        self.test_dataframes.append(self._read_evaluation_file(os.path.join(self.args.data_dir, f)))
                self._test_dataset = pd.concat(self.test_dataframes)
                self._test_dataset[self.sample_id_col] = range(len(self._test_dataset))
                self.logger.info("In evaluation mode. Dataset Shapes:\n\tTest: {}".format(self._test_dataset.shape))

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """ Return the argument parser with necessary args for this class appended to it """
        parser.add_argument("--data_dir", type=str, required=False,
                            help="If all of your data is in one folder, this is the path to that directory containing the csvs, or alternatively a single csv file")
        return parser