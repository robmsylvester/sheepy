import torch
import os
import argparse
import numpy as np
import wandb
import json
from tqdm import tqdm
from pytorch_lightning import LightningDataModule
from torch import nn, optim
from argparse import Namespace
from torch.nn import functional as F
from torchnlp.utils import lengths_to_mask
from lib.src.nlp.tokenizer import mask_fill
from lib.src.models.base_classifier import BaseClassifier
from lib.src.models.fully_connected_classifier import FullyConnectedClassifier
from lib.src.nlp.text_representation import TextRepresentation

class TransformerClassifier(BaseClassifier):
    """
    Sample model to show how to use a Transformer model to classify sentences.

    This model builds upon the text representation class then attaches a head from the fully connected classifier class,
    stitching them together according to your config params

    :param args: ArgumentParser containing the hyperparameters.
    :param data: LightningDataModule object containing implementations of train_dataloader, val_dataloader,
     and necessary other ETL.
    """

    def __init__(self, args: Namespace, data: LightningDataModule) -> None:
        super().__init__(args, data)

        # build model
        self._build_model()
        self._build_loss()
        if self.args.hparams['nr_frozen_epochs'] > 0:
            self.text_representation.freeze_encoder()

    def _build_model(self) -> None:
        """ Initializes the BERT model and the classification head."""
        self.text_representation = TextRepresentation(self.args.hparams['encoder_model'])

        layer_config = []

        #Build a list of hidden layer sizes starting with output size of the transformers
        input_size = output_size = self.text_representation.encoder_features
        for layer_idx, layer_size in enumerate(self.args.hparams['hidden_layer_sizes']):
            layer_config.append({
                "input_size": input_size,
                "output_size": layer_size,
                "dropout_p": self.args.hparams['dropout_p']
            })
            input_size = layer_size
        
        #Logit layer handled separately
        layer_config.append({
            "input_size": input_size,
            "output_size": self.args.hparams["num_labels"],
            "dropout_p": 0
        })

        self.classification_head = FullyConnectedClassifier(layer_config)

    def _build_loss(self):
        """ Initializes the loss function/s."""
        self.class_weights = self._get_class_weights()
        self._loss = nn.CrossEntropyLoss(weight=self.class_weights)

    # TODO - maybe eliminate this. seems not to be called
    # TODO: batch prediction.
    def predict(self, data_module: LightningDataModule, sample: dict) -> dict:
        """Evaluation function

        Args:
            data_module (LightningDataModule): module with method prepare_sample()
            sample (dict): Dictionary with correct key that specifies text column and value as text we want to classify

        Returns:
            dict: Dictionary with the input text and the predicted label.
        """

        if self.training:
            self.eval()

        with torch.no_grad():
            model_input, _ = data_module.prepare_sample([sample])
            model_out = self.forward(**model_input)
            logits = model_out["logits"].numpy()

            predicted_labels = [
                self.data.label_encoder.index_to_token[prediction] #TODO - this wont work
                for prediction in np.argmax(logits, axis=1)
            ]
            sample["predicted_label"] = predicted_labels[0]

        return sample

    def predict_prob(self, data_module: LightningDataModule, sample: dict) -> dict:
        """
        Predict function that returns probability

        Args:
            data_module: module with method prepare_sample()
            Sample: Dictionary with correct key that specifies text column and value as text we want to classify
        Returns:
            Dictionary with the input text and the predicted softmax label probability
        """

        if self.training:
            self.eval()

        with torch.no_grad():
            model_input, _ = data_module.prepare_sample([sample])
            model_out = self.forward(**model_input)
            softmax_p = F.softmax(model_out["logits"], axis=1).numpy()
            return softmax_p[0]

    # def evaluate_live(self, data_module: LightningDataModule):
    #     """
    #     The evaluate live method for the base transformer classifier is just a REPL that processes text and passes
    #     it to the trained model.

    #     Args:
    #         data_module (LightningDataModule): module implementing method prepare_sample()
    #     """
    #     print("Live Demo Mode.\nEnter 'q' or 'quit' (without quotes) to exit the program.\nEnter a single text_sample to run classification on.\n")
    #     while True:
    #         user_input = input("> ")
    #         if user_input == "q" or user_input == "quit":
    #             break
    #         sample = {}    # def evaluate_live(self, data_module: LightningDataModule):
    #     """
    #     The evaluate live method for the base transformer classifier is just a REPL that processes text and passes
    #     it to the trained model.

    #     Args:
    #         data_module (LightningDataModule): module implementing method prepare_sample()
    #     """
    #     print("Live Demo Mode.\nEnter 'q' or 'quit' (without quotes) to exit the program.\nEnter a single text_sample to run classification on.\n")
    #     while True:
    #         user_input = input("> ")
    #         if user_input == "q" or user_input == "quit":
    #             break
    #         sample = {}
    #         sample[data_module] = user_input.strip()
    #         prediction = self.predict_prob(data_module, sample=sample)
    #         print(prediction)

    # def evaluate_file(self, file_path: str, out_path: str = None):
    #     """
    #     Evaluates a file, with one text_sample on each file, and gets the model prediction for each line. 
    #     Sorts the return values with highest (positives) first, and appends to the out_path.

    #     Args:
    #         file_path (str): input file with one text_sample prediction done per line
    #         out_path (str): output path of sorted predictions. if none, prints formatted to stdout
    #     """
    #     with open(file_path) as fp:
    #         results_dict = {}
    #         for _, line in enumerate(tqdm(fp)):
    #             sample = {}
    #             sample[self.data.text_col] = line.strip()
    #             prediction = self.predict_prob(self.data, sample=sample)
    #             results_dict[line.strip()] = prediction
    #     sorted_results_dict = {k: str(v) for k, v in sorted(
    #         results_dict.items(), key=lambda x: x[1])}
    #     with open(out_path, "w") as fp:
    #         json.dump(sorted_results_dict, fp)
    #         sample[data_module] = user_input.strip()
    #         prediction = self.predict_prob(data_module, sample=sample)
    #         pos_prob = prediction[1]
    #         print(pos_prob)

    # def evaluate_file(self, file_path: str, out_path: str = None):
    #     """
    #     Evaluates a file, with one text_sample on each file, and gets the model prediction for each line. 
    #     Sorts the return values with highest (positives) first, and appends to the out_path.

    #     Args:
    #         file_path (str): input file with one text_sample prediction done per line
    #         out_path (str): output path of sorted predictions. if none, prints formatted to stdout
    #     """
    #     with open(file_path) as fp:
    #         results_dict = {}
    #         for _, line in enumerate(tqdm(fp)):
    #             sample = {}
    #             sample[self.data.text_col] = line.strip()
    #             prediction = self.predict_prob(self.data, sample=sample)
    #             pos_prob = prediction[1]
    #             results_dict[line.strip()] = pos_prob
    #     sorted_results_dict = {k: str(v) for k, v in sorted(
    #         results_dict.items(), key=lambda x: x[1])}
    #     with open(out_path, "w") as fp:
    #         json.dump(sorted_results_dict, fp)

    def forward(self, tokens, lengths) -> dict:
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]
        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        tokens = tokens[:, : lengths.max()]

        # When using just one GPU this should not change behavior
        # but when splitting batches across GPU the tokens have padding
        # from the entire original batch
        mask = lengths_to_mask(lengths, device=tokens.device)

        # Run BERT model.
        word_embeddings = self.text_representation.model(tokens, mask)[0]

        # Average Pooling
        word_embeddings = mask_fill(
            0.0, tokens, word_embeddings, self.data.tokenizer.pad_index
        )
        sentemb = torch.sum(word_embeddings, 1)
        sum_mask = mask.unsqueeze(-1).expand(word_embeddings.size()).float().sum(1)
        sentemb = sentemb / sum_mask

        # Linear Classifier
        return {"logits": self.classification_head(sentemb)}

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.classification_head.parameters()},
            {
                "params": self.text_representation.model.parameters(),
                "lr": self.args.hparams['encoder_learning_rate'],
            }
        ]
        optimizer = optim.Adam(
            parameters, lr=self.args.hparams['learning_rate'])
        return [optimizer], []

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.args.hparams['nr_frozen_epochs']:
            self.text_representation.unfreeze_encoder()

    # def save_onnx_model(self, save_name="model_final.onnx"):    # def save_onnx_model(self, save_name="model_final.onnx"):
    #     """
    #     Uses W&B's and Torch's Onnx hook to save the model
    #     """
    #     save_location = os.path.join(self.args.output_dir, save_name)
    #     dummy_input_dimensions = torch.zeros(
    #         self.text_representation.encoder_features, device=self.device)
    #     torch.onnx.export(self, dummy_input_dimensions, save_location)
    #     wandb.save(save_location)
    #     """
    #     Uses W&B's and Torch's Onnx hook to save the model
    #     """
    #     save_location = os.path.join(self.args.output_dir, save_name)
    #     dummy_input_dimensions = torch.zeros(
    #         self.text_representation.encoder_features, device=self.device)
    #     torch.onnx.export(self, dummy_input_dimensions, save_location)
    #     wandb.save(save_location)

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """ Return the argument parser with necessary args for this class appended to it """
        return parser
