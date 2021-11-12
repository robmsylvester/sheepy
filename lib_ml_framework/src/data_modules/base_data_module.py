import argparse
import pandas as pd
import os
import wandb
from typing import Union, List, Dict, Optional
from pytorch_lightning import LightningDataModule
from torchnlp.encoders import LabelEncoder
from torch.utils.data import DataLoader, RandomSampler
from lib_ml_framework.src.common.tokenizer import Tokenizer
from lib_ml_framework.src.common.collate import single_text_collate_function, CollatedSample
from lib_ml_framework.src.common.logger import get_std_out_logger


class BaseDataModule(LightningDataModule):

    # it's used for joining predicted values with original dataset records a batch inference (see output_dataset method)
    sample_id_col = 'sample_id' #TODO - put in init function on self

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.logger = get_std_out_logger()
        self.args = args
        self._set_tune_params()

        self.evaluate = self.args.evaluate  # readability
        self.label_col = self.args.hparams["label"]  # readability
        self.text_col = self.args.hparams["text"]  # readability

        self._train_dataset, self._val_dataset, self._test_dataset = None, None, None
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

        self.prepare_target = True if not self.evaluate else False
        self.nlp = {'tokenizer': Tokenizer(self.args.hparams['encoder_model'])}

    def _set_tune_params(self):
        # TODO - we may want to support iterating through other config than hparams, such as the encoder_model
        if self.args.tune:
            wandb.init()  # gives access to wandb.config
            for k, v in wandb.config.items():
                if k in self.args.hparams:
                    self.args.hparams[k] = v
                    self.logger.debug(
                        "Setting data module hyperparameter {} to sweep value {}".format(k, v))

    def prepare_data(self):
        raise NotImplementedError("This must be implemented by child class.")

    def prepare_sample(self):
        raise NotImplementedError(
            "Collate function must be implemented by child class.")
    
    def setup(self, stage: Optional[str] = None):
        pass

    def _verify_module(self):
        """
        Verify that the dataset shapes are correct, the class sizes and labels match the parameters, and if all is
        well, then we build the encoder
        """
        pass

    def save_data_module(self, out_path: str = None):
        # this class has nothing to pickle
        pass

    def load_data_module(self, in_path: str = None):
        # this class has nothing to pickle
        pass

    def nlp(self):
        """Initializes nlp tools on the class"""
        # TODO - move this to arg on automodel
        self.nlp = {'tokenizer': Tokenizer("bert-base-uncased")}

    def prepare_sample(self, sample: list) -> CollatedSample:
        """
        Function that prepares a sample to input the model.
        args:
            sample - list of dictionaries that contain the specified values that you will use to transform the sample into a inputs to
                the neural network. This sample will contain the text input column that will be encoded as one of the dictionaries,
                and probably a few others, such as a label key. The length of sample will be the batch size of your model, which then
                will be divided equally over your GPU's

        Returns:
            - dictionary with the expected model inputs, with each of the inputs in this case containing a key-value pair specifying which of
                the text inputs are included (context windows, meeting name, etc.)
            - dictionary with the expected target labels.
        """
        if not hasattr(self, "nlp"):
            raise ValueError(
                "Missing attribute nlp on data module object. It is likely the nlp() method has not been called.")
        if self.text_col is None:
            raise NotImplementedError(
                "To use the default collate function you need a text column.\nAlternatively, write one for the x_cols to prepare your data")

        # TODO - turn this into always using windowed text collate function
        return single_text_collate_function(sample,
                                            self.text_col,
                                            self.label_col,
                                            self.sample_id_col,
                                            self.nlp['tokenizer'],
                                            self.label_encoder,
                                            prepare_target=self.prepare_target,
                                            prepare_sample_id=self.evaluate)

    def write_predictions_to_disk(self, predictions_df: pd.DataFrame) -> None:
        """
        Joins predictions df to the self._test_dataset on and sample_id_col
        and writes the result to the output dir
        :param predictions_df: pandas dataframe with predicted values
        :return: none
        """
        df = self._test_dataset.set_index(self.sample_id_col).join(
            predictions_df.set_index(self.sample_id_col))

        self._write_predictions_to_disk(df)

    def _write_predictions_to_disk(self, predictions_df: pd.DataFrame) -> None:
        raise NotImplementedError(
            "_write_predictions_to_disk function must be implemented by child class.")
    
    def _verify_dataset_record_type(self, dataset: Union[pd.DataFrame, List[Dict]]) -> List[Dict]:
        """Verifies that dataset returns records datatype of type list[dict], converting from
        pandas if necessary.

        Args:
            dataset (Optional[pd.DataFrame, List[Dict]]): Dataset

        Returns:
            List[Dict]: Records of dataset as list of dictionaries
        """
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.to_dict("records")
        assert isinstance(dataset, list) and isinstance(dataset[0], dict)
        return dataset

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        self._train_dataset = self._verify_dataset_record_type(self._train_dataset)
        return DataLoader(
            dataset=self._train_dataset,
            sampler=RandomSampler(self._train_dataset),
            batch_size=self.args.hparams['batch_size'],
            collate_fn=self.prepare_sample,
            num_workers=self.args.hparams['loader_workers'],
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._val_dataset = self._verify_dataset_record_type(self._val_dataset)
        return DataLoader(
            dataset=self._val_dataset,
            batch_size=self.args.hparams['batch_size'],
            collate_fn=self.prepare_sample,
            num_workers=self.args.hparams['loader_workers'],
        )

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the test set. """
        self._test_dataset = self._verify_dataset_record_type(self._test_dataset)
        return DataLoader(
            dataset=self._test_dataset ,
            batch_size=self.args.hparams['batch_size'],
            collate_fn=self.prepare_sample,
            num_workers=self.args.hparams['loader_workers'],
        )

    # TODO - probably move to csv, generalize one
    def _build_label_encoder(self):
        """ Builds out custom label encoder to specify l
        ogic for which outputs will be in logits layer
        """
        if not isinstance(self._train_dataset, pd.DataFrame):
            raise NotImplementedError(
                "Currently the default label encoder function only supports pandas dataframes")
        train_labels_list = self._train_dataset[self.label_col].unique(
        ).tolist()
        assert len(train_labels_list) == self.args.hparams["num_labels"], "Passed {} to num_labels arg but see {} unique labels in train dataset.\nLabels are {}".format(
            self.args.hparams["num_labels"], len(train_labels_list), train_labels_list)
        self.label_encoder = LabelEncoder(
            train_labels_list,
            reserved_labels=[])
        # TODO - this may not be always what we want
        self.label_encoder.unknown_index = 0
        self.logger.info("\nEncoded Labels:\n{}".format(
            self.label_encoder.vocab))
        assert self.label_encoder.vocab_size == self.args.hparams["num_labels"]

    # TODO - probably move to csv, generalize one
    def _set_class_sizes(self):
        self.train_class_sizes = None  # Write this
        raise NotImplementedError("This must be implemented by child class.")

    @ classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """ Return the argument parser with necessary args for this class appended to it """
        return parser
