from collections import namedtuple
from lib.src.nlp.label_encoder import LabelEncoder
from lib.src.nlp.tokenizer import Tokenizer
from torchnlp.utils import collate_tensors
from torch import Tensor
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
            elif isinstance(label_keys, str):
                targets = {"labels": label_encoder.batch_encode(sample[label_keys])}
        except KeyError:
            raise KeyError(
                "Label key {} does not exist in sample. Sample keys are:\n{}".format(label_keys, list(sample.keys())))
        except RuntimeError:
            raise Exception("Label encoder found an unknown label.")

    return CollatedSample(inputs, targets, ids)