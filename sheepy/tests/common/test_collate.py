import pytest
import torch
from torchnlp.utils import collate_tensors

from sheepy.src.common.collate import _concatenate_text_samples, single_text_collate_function
from sheepy.src.nlp.label_encoder import LabelEncoder
from sheepy.src.nlp.tokenizer import Tokenizer


@pytest.fixture
def batch_sample() -> list:
    batch = [
        {
            'text': "first",
            'label': 0,
            'sample_id': 1
        },
        {
            'text': "second",
            'label': 1,
            'sample_id': 2
        }
    ]
    return batch


@pytest.fixture
def windowed_batch_sample() -> list:
    batch = [
        {
            'text': "fourth",
            'text_prev_1': 'third!',
            'text_prev_2': 'second',
            'text_prev_3': 'first',
            'text_next_1': 'fifth',
            'text_next_2': 'sixth',
            'text_next_3': 'seventh',
            'additional': 'tomato',
            'additional2': 'floop',
            'label': 0,
            'sample_id': 1
        },
        {
            'text': "fifth",
            'text_prev_1': 'fourth',
            'text_prev_2': 'third!',
            'text_prev_3': 'second',
            'text_next_1': 'sixth',
            'text_next_2': 'seventh',
            'text_next_3': 'eighth',
            'additional': 'potato',
            'additional2': 'bloop',
            'label': 1,
            'sample_id': 2
        }
    ]
    return batch

@pytest.fixture
def multilabel_batch_sample() -> list:
    batch = [
        {
            'text': "first",
            'label1': 0,
            'label2': 0,
            'label3': 0,
            'label4': 0,
            'sample_id': 1
        },
        {
            'text': "second",
            'label1': 0,
            'label2': 1,
            'label3': 0,
            'label4': 1,
            'sample_id': 2
        }
    ]
    return batch


@pytest.fixture
def multilabel_windowed_batch_sample() -> list:
    batch = [
        {
            'text': "fourth",
            'text_prev_1': 'third!',
            'text_prev_2': 'second',
            'text_prev_3': 'first',
            'text_next_1': 'fifth',
            'text_next_2': 'sixth',
            'text_next_3': 'seventh',
            'additional': 'tomato',
            'additional2': 'floop',
            'label1': 0,
            'label2': 0,
            'label3': 0,
            'label4': 0,
            'sample_id': 1
        },
        {
            'text': "fifth",
            'text_prev_1': 'fourth',
            'text_prev_2': 'third!',
            'text_prev_3': 'second',
            'text_next_1': 'sixth',
            'text_next_2': 'seventh',
            'text_next_3': 'eighth',
            'additional': 'potato',
            'additional2': 'bloop',
            'label1': 0,
            'label2': 1,
            'label3': 0,
            'label4': 1,
            'sample_id': 2
        }
    ]
    return batch

@pytest.fixture
def sample_windowed_batch_sample() -> list:
    batch = [
        {
            'text': "fourth",
            'text_prev_1': 'third',
            'text_prev_2': 'second',
            'text_prev_3': 'first',
            'text_next_1': 'fifth',
            'text_next_2': 'sixth',
            'text_next_3': 'seventh',
            'label': 0,
            'sample_id': 1
        },
        {
            'text': "fifth",
            'text_prev_1': 'fourth',
            'text_prev_2': 'third',
            'text_prev_3': 'second',
            'text_next_1': 'sixth',
            'text_next_2': 'seventh',
            'text_next_3': 'eighth',
            'label': 0,
            'sample_id': 2
        }
    ]
    return batch

@pytest.fixture
def label_encoder() -> LabelEncoder:
    return LabelEncoder(
        [0, 1],
        unknown_label=None)

@pytest.fixture
def multilabel_label_encoder() -> LabelEncoder:
    return LabelEncoder(
        ["label1","label2","label3","label4"],
        multilabel=True,
        unknown_label=None)

@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer("bert-base-uncased")


def collect_tensors(container) -> list:
    collected = []
    if isinstance(container, tuple):
        for elem in container:
            collected += collect_tensors(elem)
    elif isinstance(container, torch.Tensor):
        collected.append(container)
    elif isinstance(container, dict):
        for val in container.values():
            collected += collect_tensors(val)
    else:
        raise ValueError('unexpected value type {}'.format(type(container)))
    return collected


def collect_keys(container) -> list:
    collected = []
    if isinstance(container, tuple):
        for elem in container:
            collected += collect_keys(elem)
    elif isinstance(container, dict):
        collected += container.keys()
        for key in container.keys():
            collected += collect_keys(key)
    return collected


def test_single_label_single_text_collate(batch_sample, tokenizer, label_encoder):
    collated = single_text_collate_function(
        batch_sample,
        'text',
        'label',
        'sample_id',
        tokenizer,
        label_encoder=label_encoder,
        evaluate=False
    )

    tensors = collect_tensors(collated)

    assert (len(tensors) == 3) #inputs tokens, input lengths, targets, no ids
    assert (all(elem.shape[0] == len(batch_sample) for elem in iter(tensors)))

    #Test tokens match
    assert torch.all(torch.eq(tensors[0][0], torch.Tensor([101,2034,102]))) #2034 = symbol for 'first'
    assert torch.all(torch.eq(tensors[0][1], torch.Tensor([101,2117,102])))

    #Test lengths match
    assert torch.all(torch.eq(tensors[1], torch.Tensor([3,3])))

    #Test labels match
    assert torch.all(torch.eq(tensors[2], torch.Tensor([0,1])))


def test_multi_label_single_text_collate(multilabel_batch_sample, tokenizer, multilabel_label_encoder):
    collated = single_text_collate_function(
        multilabel_batch_sample,
        'text',
        ['label1','label2','label3','label4'],
        'sample_id',
        tokenizer,
        label_encoder=multilabel_label_encoder,
        evaluate=False
    )

    tensors = collect_tensors(collated)

    assert (len(tensors) == 3) #inputs tokens, input lengths, targets, no ids
    assert (all(elem.shape[0] == len(multilabel_batch_sample) for elem in iter(tensors)))

    #Test tokens match
    assert torch.all(torch.eq(tensors[0][0], torch.Tensor([101,2034,102]))) #2034 = symbol for 'first'
    assert torch.all(torch.eq(tensors[0][1], torch.Tensor([101,2117,102])))

    #Test lengths match
    assert torch.all(torch.eq(tensors[1], torch.Tensor([3,3])))

    #Test labels match
    assert torch.all(torch.eq(tensors[2][0], torch.Tensor([0,0,0,0])))
    assert torch.all(torch.eq(tensors[2][1], torch.Tensor([0,1,0,1])))

def test_concatenate_text_samples(windowed_batch_sample):
    collated = collate_tensors(windowed_batch_sample)
    concatenated = _concatenate_text_samples(
        collated,
        ['text_prev_3', 'text_prev_2', 'text_prev_1', 'text', 'text_next_1', 'text_next_2', 'text_next_3'],
        default_separator="."
    )

    expected = ["first. second. third! fourth. fifth. sixth. seventh.",
        "second. third! fourth. fifth. sixth. seventh. eighth."]

    assert concatenated[0] == expected[0]
    assert concatenated[1] == expected[1]
