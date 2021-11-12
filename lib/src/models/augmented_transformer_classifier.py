import torch
from typing import Dict
from argparse import ArgumentParser, Namespace
from torchnlp.utils import lengths_to_mask
from pytorch_lightning import LightningDataModule
from lib.src.common.tokenizer import mask_fill
from lib.src.common.collate import round_size
from lib.src.models.base_transformer_classifier import TransformerClassifier
from lib.src.models.fully_connected_classifier import FullyConnectedClassifier
from lib.src.nlp.text_representation import TextRepresentation

class AugmentedTransformerClassifier(TransformerClassifier):
    """
    This model builds upon the text representation class then attaches a head from the fully connected classifier class,
    stitching them together according to your config params.

    The difference between this transformer and the other one is that it has other features concatenated, such as topics,
    as well as possibly other text representations

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

    def _compute_input_layer_size(self, num_encoder_features: int):
        n_prev_text_samples = self.args.hparams['n_prev_text_samples'] if self.args.hparams['n_prev_text_samples_sample_size'] is None else self.args.hparams['n_prev_text_samples_sample_size']
        n_next_text_samples = self.args.hparams['n_next_text_samples'] if self.args.hparams['n_next_text_samples_sample_size'] is None else self.args.hparams['n_next_text_samples_sample_size']
        num_context_text_samples = 1 if self.args.hparams['text_sample_concatenation'] else n_prev_text_samples + n_next_text_samples
        num_title_embedding_features = 1 if self.args.hparams['document_title'] else 0
        expected_size = (num_context_text_samples + num_title_embedding_features + 1) * num_encoder_features
        return expected_size

    def _build_model(self) -> None:
        """Initializes the BERT model and the classification head."""

        self.text_representation = TextRepresentation(
            self.args.hparams['encoder_model'])

        layer_config = []
        input_size = output_size = self._compute_input_layer_size(self.text_representation.encoder_features)
        for layer_idx in range(self.args.hparams['num_dense_layers']):
            output_size = int(output_size * self.args.hparams['dense_layer_scale_factor'])
            output_size = round_size(output_size)
            layer_config.append({
                "input_size": input_size,
                'output_size': output_size,
                "dropout_p": self.args.hparams['dropout_p']
            })
            input_size = output_size

        layer_config[-1]['output_size'] = self.args.hparams["num_labels"]
        layer_config[-1]['dropout_p'] = 0

        self.classification_head = FullyConnectedClassifier(layer_config)
    
    def forward(self, tokens: Dict, lengths: Dict, extra_features: Dict) -> Dict:
        """PyTorch forward function for the neural network classifier

        Args:
            tokens: Dictionary of text features, where each feature contains the source tokens of that feature. 
                So each key in this dictionary will be a text column name, and each value will be a tensor of [batch_size x src_seq_len]
            
            lengths: Dictionary of text features, where each feature contains the source lengths of the tokens for that feature. 
                So each key in this dictionary will be a text column name, and each value will be a tensor of [batch_size]
        
            extra_features: Dictionary of text features, where each feature contains a column name for the name of the extra features
                So each key in this dictionary will be a text column name, and each value will be a tensor of [batch_size x feature_size] where
                feature_size is the size of that feature. In the cases where it is a single float, the tensor will be of shape [batch_size]
        
        Returns:
            Dictionary with model outputs (e.g: logits)
        """

        text_embeddings = []
        text_inputs = sorted(list(tokens.keys()))

        #TODO - this needs to be an ordered dict
        #Go through each of the text inputs, pass them through the same transformer
        for text_input in text_inputs:
            t = tokens[text_input]
            l = lengths[text_input]

            t = t[:, : l.max()]

            # When using just one GPU this should not change behavior
            # but when splitting batches across GPU the tokens have padding
            # from the entire original batch
            mask = lengths_to_mask(l, device=t.device)

            # Run transformer model
            word_embeddings = self.text_representation.model(t, mask)[0]

            # Average Pooling
            word_embeddings = mask_fill(
                0.0, t, word_embeddings, self.data.nlp['tokenizer'].pad_index
            )
            text_embedding = torch.sum(word_embeddings, 1)
            sum_mask = mask.unsqueeze(-1).expand(word_embeddings.size()
                                                ).float().sum(1)
            text_embedding = text_embedding / sum_mask
            text_embeddings.append(text_embedding)
        

        all_text_embeddings = torch.cat(text_embeddings, dim=1)

        if not len(extra_features):
            return {"logits": self.classification_head(all_text_embeddings)}
        
        all_extra_features = torch.cat([v for v in extra_features.values()], dim=1)
        feature_vector = torch.cat([all_text_embeddings, all_extra_features], dim=1)

        return {"logits": self.classification_head(feature_vector)}