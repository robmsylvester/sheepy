import argparse
import os
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
import wandb
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler

from sheepy.common.collate import CollatedSample, single_text_collate_function
from sheepy.common.logger import get_std_out_logger
from sheepy.nlp.label_encoder import LabelEncoder
from sheepy.nlp.tokenizer import Tokenizer


class BaseDataModule(LightningDataModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.logger = get_std_out_logger()
        self.args = args
        self._set_tune_params()
        self.sample_id_col = "sample_id"
        self.evaluate = self.args.evaluate  # readability
        self.label_col = self.args.label  # readability
        self.text_col = self.args.text  # readability

        self._train_dataset, self._val_dataset, self._test_dataset, self._predict_dataset = (
            None,
            None,
            None,
            None,
        )
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

        self.tokenizer = Tokenizer(self.args.encoder_model)

    def _set_tune_params(self):
        # TODO - we may want to support iterating through other config than hparams, such as the encoder_model
        if self.args.tune:
            wandb.init()  # gives access to wandb.config
            for k, v in wandb.config.items():
                if k in self.args:
                    self.args[k] = v
                    self.logger.debug(
                        "Setting data module hyperparameter {} to sweep value {}".format(k, v)
                    )

    def prepare_data(self):
        raise NotImplementedError("This must be implemented by child class.")

    def setup(self, stage: Optional[str] = None):
        pass

    def teardown(self, stage: Optional[str] = None):
        pass

    def _verify_module(self):
        """
        Verify that the dataset shapes are correct, the class sizes and labels match the parameters.
        """
        pass

    def save_data_module(self, out_path: str = None):
        pass

    def load_data_module(self, in_path: str = None):
        pass

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
        if self.text_col is None:
            raise NotImplementedError(
                "To use the default collate function you need a text column.\nAlternatively, write one for the x_cols to prepare your data"
            )

        # TODO - turn this into always using windowed text collate function
        return single_text_collate_function(
            sample,
            self.text_col,
            self.label_col,
            self.sample_id_col,
            self.tokenizer,
            self.label_encoder,
            evaluate=self.evaluate,
        )

    def _write_predictions(self, outputs: List[dict]) -> None:
        all_columns = self.label_encoder.vocab
        all_columns.append(self.sample_id_col)
        all_batch_predictions_df = pd.DataFrame(columns=all_columns)

        for output in outputs:
            prediction_logits = output["logits"].cpu().squeeze()
            sample_ids = pd.Series(output["sample_id_keys"].cpu().squeeze()).astype("int")
            prediction_softmax = torch.nn.Softmax(dim=1)(prediction_logits)
            batch_prediction_df = pd.DataFrame(prediction_softmax).astype("float")
            batch_prediction_df = batch_prediction_df.rename(
                columns={
                    label_idx: label for label_idx, label in enumerate(self.label_encoder.vocab)
                }
            )
            batch_prediction_df[self.sample_id_col] = sample_ids
            all_batch_predictions_df = all_batch_predictions_df.append(
                batch_prediction_df, ignore_index=True
            )

        all_batch_predictions_df = all_batch_predictions_df.reset_index(drop=True)
        cols = all_batch_predictions_df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        all_batch_predictions_df = all_batch_predictions_df[cols]

        self.logger.info(
            "Writing CSV file with {} predictions to {}".format(
                all_batch_predictions_df.shape[0], self.args.output_prediction_path
            )
        )
        all_batch_predictions_df.to_csv(self.args.output_prediction_path)

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

    def _load_text_from_text_file(self, filepath: str) -> List[dict]:
        """[summary]

        Args:
            filepath (str): [description]

        Returns:
            List[dict]: [description]
        """
        prepared_inputs = []
        with open(filepath, "r") as f:
            for idx, line in enumerate(f):
                line = line.rstrip()
                prepared_input = {
                    self.text_col: line,
                    self.label_col: None,
                    self.sample_id_col: idx,
                }
                prepared_inputs.append(prepared_input)
        return prepared_inputs

    def _load_text_from_raw_input(self) -> List[dict]:
        sample_text = input("Enter sample text. (Press Ctrl+C to exit)\n")
        prepared_input = {self.text_col: sample_text, self.label_col: None, self.sample_id_col: 0}
        return [prepared_input]

    def train_dataloader(self) -> DataLoader:
        """Function that loads the train set."""
        self._train_dataset = self._verify_dataset_record_type(self._train_dataset)
        return DataLoader(
            dataset=self._train_dataset,
            sampler=RandomSampler(self._train_dataset),
            batch_size=self.args.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.args.loader_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Function that loads the validation set."""
        self._val_dataset = self._verify_dataset_record_type(self._val_dataset)
        return DataLoader(
            dataset=self._val_dataset,
            batch_size=self.args.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.args.loader_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Function that loads the test set."""
        self._test_dataset = self._verify_dataset_record_type(self._test_dataset)
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.args.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.args.loader_workers,
        )

    def predict_batch_dataloader(self) -> DataLoader:
        """Function that loads batch prediction by processing lines in a text file"""
        self._predict_dataset = self._load_text_from_text_file(self.args.evaluate_batch_file)
        return DataLoader(
            dataset=self._predict_dataset,
            batch_size=self.args.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.args.loader_workers,
        )

    def predict_live_dataloader(self) -> DataLoader:
        """Function that loads the batch prediction set by processing a single piece of text input to the shell"""
        self._predict_dataset = self._load_text_from_raw_input()
        return DataLoader(
            dataset=self._predict_dataset,
            batch_size=self.args.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.args.loader_workers,
        )

    def _build_label_encoder(self):
        self.label_encoder = LabelEncoder.initializeFromDataframe(
            self._train_dataset, self.args.label
        )
        self.logger.info("Label Encoder Vocab: {}".format(self.label_encoder.vocab))

    # TODO - probably move to csv, generalize one
    def _set_class_sizes(self):
        self.train_class_sizes = None  # Write this
        raise NotImplementedError("This must be implemented by child class.")

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Return the argument parser with necessary args for this class appended to it"""
        return parser
