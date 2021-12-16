import pandas as pd
from torch import Tensor, tensor, stack, long
from typing import List

class LabelEncoder():
    def __init__(self, labels: List[str], multilabel: bool=False, unknown_label: str=None):
        self.multilabel = multilabel
        self.unknown_label = unknown_label
        self.vocab = labels
        self.labels_to_index = {val: idx for idx, val in enumerate(self.vocab)}
        self.index_to_labels = {idx: val for idx, val in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
    
    def encode(self, label_str: str) -> Tensor:
        """Given a single label, return a tensor of shape [1] containing the integer representation of the label

        Args:
            label (str): Label that must exist in the vocabulary of the label encoder, unless unknown_label is set

        Returns:
            Tensor: Single integer representation of the label
        
        Raises:
            ValueError: label does not exist in the vocabulary and there is not an unknown_label argument set
        """
        if label_str not in self.labels_to_index:
            if self.unknown_label is None:
                raise ValueError("No unknown label is set in LabelEncoder and came across unknown label {}".format(label_str))
            else:
                return tensor(self.labels_to_index[self.unknown_label], dtype=long)

        return tensor(self.labels_to_index[label_str], dtype=long)
    
    def batch_encode(self, label_strings: List[str]) -> Tensor:
        """Given a list of labels, return a tensor of shape [batch_size] containing the integer representation of the labels

        Args:
            label_strings (List[str]): Labels that must exist in the vocabulary of the label encoder, unless unknown_label is set.

        Returns:
            Tensor: 1D tensor of size batch_size, equal to len(label_str), with the integer representation of each label
        
        Raises:
            ValueError: any of the labels do not exist in the vocabulary and there is not an unknown_label argument set
        """
        batch = [self.encode(label_str) for label_str in label_strings]
        return stack(batch, dim=0)

    def decode(self, label_tensor: Tensor) -> str:
        """Given a tensor with a single integer label, return a tensor of shape [1] containing the string representation of the label.
        No error/type checking is done here so only pass through to the decoder what was passed through the encoder first.

        Args:
            label_tensor (Tensor): Tensor containing integer represenation that must exist in the vocabulary of the label encoder, unless unknown_label is set

        Returns:
            str: Single string representation of the label
        """
        label_int = label_tensor.squeeze().item()
        return self.index_to_labels[label_int]
    
    def batch_decode(self, label_tensor: Tensor) -> List[str]:
        """Given a tensor with a multiple integer labels, return a tensor of shape [batch_size] containing the string representations of the labels

        Args:
            label_tensor (Tensor): Tensor containing integer represenations that must exist in the vocabulary of the label encoder, unless unknown_label is set

        Returns:
            List[str]: List of strings of length batch_size, equal to len(label_str), with the string representation of each label
        
        Raises:
            ValueError: any label integer index does not exist in the vocabulary (is larger than vocab size) and there is not an unknown_label argument set
        """
        batch = [t.squeeze(0) for t in label_tensor.split(1, dim=0)]
        return [self.decode(label_int) for label_int in batch]
    
    @classmethod
    def initializeFromDataframe(cls, df: pd.DataFrame, label_col: str, reserved_labels: List[str] = [], unknown_label: str=None):
        labels_list = reserved_labels
        for found_label in df[label_col].unique().tolist():
            if found_label not in labels_list:
                labels_list.append(found_label)
        
        if unknown_label is not None and unknown_label not in labels_list:
            labels_list.append(unknown_label)

        print("creating label_list {}".format(labels_list))
        return cls(labels_list, multilabel=False, unknown_label=unknown_label)
    
    @classmethod
    def initializeFromMultilabelDataframe(cls, df: pd.DataFrame, label_list: List[str], reserved_labels: List[str] = [], unknown_label: str=None):
        labels_list = reserved_labels
        for label in label_list:
            unique_vals = df[label].unique()
            if len(unique_vals) > 2:  # this restriction can probably be removed eventually
                raise ValueError("Label {} must be binary. Instead, see values {}".format(label, str(unique_vals)))
            if label not in labels_list:
                labels_list.append(label)
        
        if unknown_label is not None and unknown_label not in labels_list:
            labels_list.append(unknown_label)
        
        return cls(labels_list, multilabel=True, unknown_label=unknown_label)


# >>> samples = ['label_a', 'label_b']
# >>> encoder = LabelEncoder(samples, reserved_labels=['unknown'], unknown_index=0)
# >>> encoder.encode('label_a')
# tensor(1)
# >>> encoder.decode(encoder.encode('label_a'))
# 'label_a'
# >>> encoder.encode('label_c')
# tensor(0)
# >>> encoder.decode(encoder.encode('label_c'))
# 'unknown'
# >>> encoder.vocab
# ['unknown', 'label_a', 'label_b']