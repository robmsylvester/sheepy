from collections import namedtuple
from lib.src.nlp.label_encoder import LabelEncoder
from lib.src.nlp.tokenizer import Tokenizer
from torchnlp.utils import collate_tensors
from torch import Tensor, FloatTensor, transpose
from typing import List, Union
import random

# A batch sample to input the model.
#     inputs: samples and their lengths
#     targets: labels
#     ids: sample ids (used to align predicted values with corresponding samples in multi-GPU environment)     
CollatedSample = namedtuple(
    'CollatedSample', ['inputs', 'targets', 'ids'])

def _get_sample_list(samples: List[str], sample_size: int=None) -> List[str]:
    """"Returns a list of samples from a list, without repetition.

    Args:
        samples (List[str]): a list of strings
        sample_size (int, optional): the number of strings to sample. If none defaults to len(samples)

    Returns:
        List[str]: the alphabetized list of sampled strings
    
    Raises:
        ValueError if the sample_size > len(samples)
    """
    if sample_size is None:
        sample_size = len(samples)

    if sample_size > len(samples):
        raise ValueError("Sampling {} samples isn't possible when only {} are available".format(sample_size, len(samples)))

    chosen = random.sample(samples, k=sample_size)
    return chosen

def _concatenate_text_samples(sample: dict, keys: List[str], default_separator: str=".") -> List[str]:
    """Given a sample dicitonary, return a single string of concatenated strings, each of which is a key
    in the list 'keys' for the dictionary. the dictionary, at each of these keys, will have a list of values
    that is as long as the batch length in the current iteration.

    Whitespace is trimmed if necessary so that a single space follows the default separator.

    Args:
        sample (dict): dictionary sample that is the output of collate_tensors().
        keys (List[str]): list of keys, already in order, to be grabbed and concatenated
        default_separator (str): separator between keys if they dont already end in "?" "!" or "."
    
    Returns:
        List([str]) -> A list of concatenated strings of length equal to len(sample_list)
    """
    out_strings = []
    sample_batch_size = len(sample[keys[0]])
    for idx in range(sample_batch_size):
        formatted_sample_string = []
        sample_strings = [sample[key][idx] for key in keys] #returns a list of length batch_size
        for ss in sample_strings:
            if len(ss):
                if ss[-1] not in ['?','!','.']:
                    ss += default_separator
                formatted_sample_string.append(ss)
        out_strings.append(" ".join([fss for fss in formatted_sample_string]))
    return out_strings

def round_size(size: int):
    """
    Rounds a neural network output layer size to the smallest power of two greater than the target number.
    Layers behave better (faster) this way.
    """
    lowest_power_of_two = 2
    while lowest_power_of_two < size:
        lowest_power_of_two *= 2
    return lowest_power_of_two

def single_text_collate_function(sample: list,
                                 text_key: str,
                                 label_keys: Union[str, List[str]],
                                 id_key: str,
                                 tokenizer: Tokenizer,
                                 label_encoder: LabelEncoder = None,
                                 evaluate: bool = False) -> CollatedSample:
    """
    Function that prepares a sample to input the model.

    Arguments:
        sample (list): list of dictionaries, one of which contains the text input column that will be encoded
        text_key (str): the key of the dictionaries in sample that contain the text to be encoded
        label_keys (str, List[str]): the list of keys in the dictionary in sample that contains the label
        id_key (str): the id key of the sample with which to uniquely identify it.
        tokenizer (Tokenizer): some tokenizer to use to encode the text, from the transformers tokenizer class for now
        label_encoder (LabelEncoder, optional): an instance of LabelEncoder.
        evaluate (bool, optional): In training mode, prepares the target from labels. Defaults to False.

    Raises:
        ValueError: The label encoder is not None with a list of labels passed
        Exception: An unnknown label is found in the label encoder

    Returns:
        CollatedSample: tuple of dictionaries with the expected model inputs, target labels, and sample ids
    """
    sample = collate_tensors(sample)
    try:
        # batch-size is first dimension in these tensors
        tokens, lengths = tokenizer.batch_encode(sample[text_key])
    except KeyError:
        raise KeyError(
            "Text key {} does not exist in sample. Sample keys are:\n{}".format(text_key, list(sample.keys())))

    inputs = {"tokens": tokens, "lengths": lengths}
    targets = {}
    ids = {}

    if evaluate:
        ids = {"sample_id_keys": Tensor(sample[id_key])}
    else:
        try:
            if isinstance(label_keys, list):
                targets = {"labels": label_encoder.batch_encode_multilabel(sample)}
                # outputs = []
                # for l in label_keys:
                #     integerized_label = [int(i) for i in sample[l]]
                #     outputs.append(integerized_label)
                # # transpose gets us to (batch_size, num_labels)
                # labels = transpose(FloatTensor(outputs), 0, 1)
                # targets = {"labels": labels}
            elif isinstance(label_keys, str):
                targets = {"labels": label_encoder.batch_encode(sample[label_keys])}
        except KeyError:
            raise KeyError(
                "Label key {} does not exist in sample. Sample keys are:\n{}".format(label_keys, list(sample.keys())))
        except RuntimeError:
            raise Exception("Label encoder found an unknown label.")

    return CollatedSample(inputs, targets, ids)


def windowed_text_collate_function(sample: list,
                                    text_key: str,
                                    label_keys: Union[str,List[str]],
                                    id_key: str,
                                    tokenizer: Tokenizer,
                                    label_encoder: LabelEncoder = None,
                                    additional_text_cols: List[str] = [],
                                    prev_sample_size: int=None,
                                    next_sample_size: int=None,
                                    concatenate: bool=False,
                                    evaluate: bool = False) -> CollatedSample:
    """Function that prepares a sample to input the model, with an optional number of windows text columns
    that will be used in the model as well.

    Args:
        sample (list): list of dictionaries, one of which contains the text input column that will be encoded
        text_key (str): the key of the dictionaries in sample that contain the text to be encoded
        label_key (str, List[str]): the list of keys in the dictionary in sample that contains the label
        id_key (str): the id key of the sample with which to uniquely identify it.
        label_encoder (LabelEncoder, optional): an instance of LabelEncoder. Defaults to None.
        tokenizer (Tokenizer): some tokenizer to use to encode the text, from the transformers tokenizer class for now
        additional_text_cols (List[str], optional): other text columns that need to be tokenized. Defaults to [].
        prev_sample_size (int, optional): number of previous text columns to sample out of those available. Defaults to None, which means all of them
        next_sample_size (int, optional): number of next text columns to sample out of those available. Defaults to None, which means all of them
        concatenate (bool, optional): If true, creates a single piece of text with the windowed text_samples, as opposed to one for each text_sample. Defaults to False.
        evaluate (bool, optional): In training mode, prepares the target from labels. Defaults to False.

    Raises:
        ValueError: The prev_sample_size > number of prev columns, or next_sample_size > num next columns
        KeyError: The label key does not exist in the label encoder
        Exception: An unnknown label is found in the label encoder

    Returns:
        CollatedSample: tuple of dictionaries with the expected model inputs, target labels, and sample ids
    """

    inputs = {'tokens': {}, 'lengths': {}}
    sample = collate_tensors(sample)
    prev_text_cols = [l for l in sample.keys() if "{}_prev_".format(text_key) in l]
    next_text_cols = [l for l in sample.keys() if "{}_next_".format(text_key) in l]

    if evaluate: #if eval, we just get the last n previous text_samples and first n next text_samples. every time
        prev_text_cols.sort(reverse=True)
        next_text_cols.sort()
        prev_text_cols = prev_text_cols[-prev_sample_size:]
        next_text_cols = next_text_cols[:next_sample_size]

    else: #if training, we sample n previous text_samples and n next text_samples, then order them
        prev_text_cols = _get_sample_list(prev_text_cols, prev_sample_size)
        next_text_cols = _get_sample_list(next_text_cols, next_sample_size)

        # We want the context in order spoken, ie, prev_3, prev_2, prev_1, but next_1, next_2, next-3
        prev_text_cols.sort(reverse=True)
        next_text_cols.sort() 

    try:
        tokens, lengths = tokenizer.batch_encode(sample[text_key])
    except KeyError:
        raise KeyError(
            "Text key {} does not exist in sample. Sample keys are:\n{}".format(text_key, list(sample.keys())))

    # first get the text_sample column itself
    tokens, lengths = tokenizer.batch_encode(sample[text_key])
    inputs['tokens'][text_key] = tokens
    inputs['lengths'][text_key] = lengths

    # then get all the additional text columns
    for col in additional_text_cols:
        tokens, lengths = tokenizer.batch_encode(sample[col])
        inputs['tokens'][col] = tokens
        inputs['lengths'][col] = lengths

    # then get all the context columns
    if concatenate: #if concatenating, we just put them together in the order prev, text, next
        ordered_text_cols = prev_text_cols + [text_key] + next_text_cols
        single_text_sample_list = _concatenate_text_samples(sample, ordered_text_cols)
        tokens, lengths = tokenizer.batch_encode(single_text_sample_list)
        inputs['tokens'][text_key+"_concatenated_context"] = tokens
        inputs['lengths'][text_key+"_concatenated_context"] = lengths
    else:
        for col in prev_text_cols + next_text_cols:
            tokens, lengths = tokenizer.batch_encode(sample[col])
            inputs['tokens'][col] = tokens
            inputs['lengths'][col] = lengths

    targets = {}
    ids = {}

    #MULTILABEL
    if evaluate:
        ids = {"sample_id_keys": Tensor(sample[id_key])}
    else:
        try:
            if isinstance(label_keys, list):
                targets = {"labels": label_encoder.batch_encode_multilabel(sample)}

                # outputs = []
                # for l in label_keys:
                #     integerized_label = [int(i) for i in sample[l]]
                #     outputs.append(integerized_label)
                # # transpose gets us to (batch_size, num_labels)
                # labels = transpose(FloatTensor(outputs), 0, 1)
                # targets = {"labels": labels}
            elif isinstance(label_keys, str):
                targets = {"labels": label_encoder.batch_encode(sample[label_keys])}
        except KeyError:
            raise KeyError(
                "Label keys {} does not match sample. Sample keys are:\n{}".format(label_keys, list(sample.keys())))
        except RuntimeError:
            raise Exception("Label encoder found an unknown label.")
    
    return CollatedSample(inputs, targets, ids)


