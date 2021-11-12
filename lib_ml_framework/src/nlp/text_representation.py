from transformers import AutoModel
import torch.nn as nn

class TextRepresentation(nn.Module):
    def __init__(self, encoder_model: str="bert"):
        super().__init__()
        self._frozen = False
        self.model = AutoModel.from_pretrained( #can be bert, albert, roberta, xlnet, etc.
            encoder_model,
            output_hidden_states=True
        )

        # set the number of features our encoder model will return...
        if encoder_model == "google/bert_uncased_L-2_H-128_A-2":
            self.encoder_features = 128
        else:
            self.encoder_features = 768
    
    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            for param in self.model.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.model.parameters():
            param.requires_grad = False
        self._frozen = True