import torch
from transformers import AutoTokenizer
from torchnlp.encoders.text.text_encoder import TextEncoder


class Tokenizer(TextEncoder):
    """
    BERT Tokenizer wrapped around the TorchNLP TextEncoder module.

    TextEncoders make it easy to go back and forth between vector representations and
    vocabulary representations. Here we add a tokenizer on top of that.
    """

    def __init__(self, pretrained_model) -> None:
        self.enforce_reversible = False
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.itos = self.tokenizer.convert_ids_to_tokens

    @property
    def unk_index(self) -> int:
        """ Returns the index used for the unknown token. """
        return self.tokenizer.unk_token_id

    @property
    def bos_index(self) -> int:
        """ Returns the index used for the begin-of-sentence token. """
        return self.tokenizer.cls_token_id

    @property
    def eos_index(self) -> int:
        """ Returns the index used for the end-of-sentence token. """
        return self.tokenizer.sep_token_id

    @property
    def pad_index(self) -> int:
        """ Returns the index used for the padding token. """
        return self.tokenizer.pad_token_id

    @property
    def vocab(self) -> list:
        """
        Returns:
            list: List of tokens in the dictionary.
        """
        return self.tokenizer.vocab

    @property
    def vocab_size(self) -> int:
        """
        Returns:
            int: Number of tokens in the dictionary.
        """
        return len(self.itos)

    def encode(self, sequence: str) -> torch.Tensor:
        """ Encodes a 'sequence'.
        Arguments:
            :param sequence: String 'sequence' to encode.

        Returns:
            torch.Tensor: encoding of the `sequence`.
        """
        sequence = TextEncoder.encode(self, sequence)
        return self.tokenizer(sequence, return_tensors="pt")["input_ids"][0]

    def batch_encode(self, sentences: list) -> (torch.Tensor, torch.Tensor):
        """
        Arguments:
            :param iterator (iterator): Batch of text to encode.
            :param **kwargs: Keyword arguments passed to 'encode'.

        Returns
            torch.Tensor, torch.Tensor: Encoded and padded batch of sequences; Original lengths of
                sequences.
        """
        tokenizer_output = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            return_length=True,
            return_token_type_ids=False,
            return_attention_mask=False,
            truncation="only_first",
            max_length=512
        )
        return tokenizer_output["input_ids"], tokenizer_output["length"]


def mask_fill(fill_value: float,
              tokens: torch.tensor,
              embeddings: torch.tensor,
              pad_index: int) -> torch.tensor:
    """
    Function that masks embeddings representing padded elements.

    Arguments:
        fill_value: the value to fill the embeddings belonging to padded tokens.
        :tokens: the input sequences [batch_size x seq_len].
        embeddings: word embeddings [batch_size x seq_len x embedding_size].
        pad_index: index of the padding token.

    Returns:
        embeddings: torch float tensor with filled masks according to fill value and pad indexes
    """
    padding_mask = tokens.eq(pad_index).unsqueeze(-1)
    return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)
