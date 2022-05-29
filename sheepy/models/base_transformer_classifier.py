import argparse
from argparse import Namespace

from pytorch_lightning import LightningDataModule
from torch import optim
from torchnlp.utils import lengths_to_mask
from transformers import AutoConfig, AutoModelForSequenceClassification

from sheepy.models.base_classifier import BaseClassifier


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

    def _build_model(self) -> None:
        """Initializes the BERT model and the classification head."""
        config = AutoConfig.from_pretrained(
            self.args.encoder_model,
            num_labels=len(self.args.label),
            id2label={id: label for id, label in enumerate(self.args.label)},
            label2id={label: id for id, label in enumerate(self.args.label)},
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.args.encoder_model,
            config=config,
        )

    def forward(self, tokens, lengths) -> dict:
        """Usual pytorch forward function.
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
        return {"logits": self.model(tokens, mask)[0]}

    def configure_optimizers(self):
        """Sets different Learning rates for different parameter groups."""
        parameters = [
            {
                "params": self.model.parameters(),
                "lr": self.args.encoder_learning_rate,
            },
        ]
        optimizer = optim.Adam(parameters, lr=self.args.learning_rate)
        return [optimizer], []

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Return the argument parser with necessary args for this class appended to it"""
        return parser
