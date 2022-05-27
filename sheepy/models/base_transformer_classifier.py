import argparse
from argparse import Namespace

import torch
from pytorch_lightning import LightningDataModule
from torch import optim
from torchnlp.utils import lengths_to_mask

from sheepy.models.base_classifier import BaseClassifier
from sheepy.models.fully_connected_classifier import FullyConnectedClassifier
from sheepy.nlp.text_representation import TextRepresentation
from sheepy.nlp.tokenizer import mask_fill


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
        input_size = self.text_representation.encoder_features
        for layer_size in self.args.hparams['hidden_layer_sizes']:
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

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """ Return the argument parser with necessary args for this class appended to it """
        return parser
