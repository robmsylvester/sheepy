from typing import Dict, List

import pandas as pd
from torch import Tensor, long, stack, tensor, transpose


class LabelEncoder:
    def __init__(self, labels: List[str], multilabel: bool = False, unknown_label: str = None):
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
                raise ValueError(
                    "No unknown label is set in LabelEncoder and came across unknown label {}".format(
                        label_str
                    )
                )
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

    # TODO - add support for unknown label?
    def batch_encode_multilabel(self, sample: Dict) -> Tensor:
        """Given a dictionary input that follows from the output of torch's collate_tensors function, create a single tensor that
        stores the labels for the label columns provided.

        Args:
            sample (Dict): Dictionary where each key is a column/feature in the dataset, which may or may not be a label, and each value
            is the concatenated batch results for this sample batch.

            Example for batch of two
            {'feature_a': torch.Size([2, 5]), 'feature_b': torch.Size([2, 5], 'label_a': torch.Size([2, 1], 'label_b': torch.Size([2, 1])}

        Returns:
            Tensor: 2D float tensor of size batch_size x num_labels

        Raises:
            ValueError: any of the labels do not exist in the vocabulary and there is not an unknown_label argument set
        """
        outputs = []
        for label in self.vocab:
            integerized_label = [int(i) for i in sample[label]]
            outputs.append(integerized_label)
        outputs = tensor(outputs)
        # transpose gets us to (batch_size, num_labels)
        labels = transpose(outputs, 0, 1)
        return labels

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

    # TODO - add support for unknown label?
    def batch_decode_multilabel(self, label_tensor: Tensor) -> Dict:
        """[summary]

        Args:
            label_tensor (Tensor): Tensor of shape (batch_size, num_labels), which is filled with 0's or 1's, where label_tensor[:, i] corresponds to the label in self.vocab[i]

        Returns:
            Dict: A dictionary of {'first_label': [0,1,1,0...], 'second_label': [1,1,0,0...]} where the keys are self.vocab and the values are numpy arrays of length batch_size
        """
        output_dict = {}
        for label_idx, label in enumerate(self.vocab):
            output_dict[label] = label_tensor[:, label_idx]
        return output_dict

        # outputs = []
        # for l in self.vocab:
        #     integerized_label = [int(i) for i in sample[l]]
        #     outputs.append(integerized_label)
        # # transpose gets us to (batch_size, num_labels)
        # labels = transpose(FloatTensor(outputs), 0, 1)
        # return labels

    @classmethod
    def initializeFromDataframe(
        cls,
        df: pd.DataFrame,
        label_col: str,
        reserved_labels: List[str] = [],
        unknown_label: str = None,
    ):
        labels_list = reserved_labels
        for found_label in df[label_col].unique().tolist():
            if found_label not in labels_list:
                labels_list.append(found_label)

        if unknown_label is not None and unknown_label not in labels_list:
            labels_list.append(unknown_label)

        print("creating label_list {}".format(labels_list))
        return cls(labels_list, multilabel=False, unknown_label=unknown_label)

    @classmethod
    def initializeFromMultilabelDataframe(
        cls,
        df: pd.DataFrame,
        label_list: List[str],
        reserved_labels: List[str] = [],
        unknown_label: str = None,
    ):
        labels_list = reserved_labels
        for label in label_list:
            unique_vals = df[label].unique()
            if len(unique_vals) > 2:  # this restriction can probably be removed eventually
                raise ValueError(
                    "Label {} must be binary. Instead, see values {}".format(
                        label, str(unique_vals)
                    )
                )
            if label not in labels_list:
                labels_list.append(label)

        if unknown_label is not None and unknown_label not in labels_list:
            labels_list.append(unknown_label)

        return cls(labels_list, multilabel=True, unknown_label=unknown_label)
