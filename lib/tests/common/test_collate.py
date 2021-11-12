import pytest
import torch
from torchnlp.encoders import LabelEncoder
from torchnlp.utils import collate_tensors
from lib.src.common.collate import single_text_collate_function, windowed_text_collate_function, _concatenate_text_samples
from lib.src.common.tokenizer import Tokenizer
from lib.src.data_modules.base_data_module import BaseDataModule as dm


@pytest.fixture
def batch_sample() -> list:
    batch = [
        {
            'text': "first",
            'label': 0,
            dm.sample_id_col: 1
        },
        {
            'text': "second",
            'label': 1,
            dm.sample_id_col: 2
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
            dm.sample_id_col: 1
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
            dm.sample_id_col: 2
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
            dm.sample_id_col: 1
        },
        {
            'text': "second",
            'label1': 0,
            'label2': 1,
            'label3': 0,
            'label4': 1,
            dm.sample_id_col: 2
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
            dm.sample_id_col: 1
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
            dm.sample_id_col: 2
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
            dm.sample_id_col: 1
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
            dm.sample_id_col: 2
        }
    ]
    return batch


@pytest.fixture
def label_encoder() -> LabelEncoder:
    return LabelEncoder(
        [0, 1],
        reserved_labels=[])


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
        dm.sample_id_col,
        tokenizer,
        label_encoder=label_encoder,
        prepare_target=True,
        prepare_sample_id=True
    )

    tensors = collect_tensors(collated)

    assert (len(tensors) == 4) #inputs tokens, input lengths, targets, ids
    assert (all(elem.shape[0] == len(batch_sample) for elem in iter(tensors)))

    #Test tokens match
    assert torch.all(torch.eq(tensors[0][0], torch.Tensor([101,2034,102]))) #2034 = symbol for 'first'
    assert torch.all(torch.eq(tensors[0][1], torch.Tensor([101,2117,102])))

    #Test lengths match
    assert torch.all(torch.eq(tensors[1], torch.Tensor([3,3])))

    #Test labels match
    assert torch.all(torch.eq(tensors[2], torch.Tensor([0,1])))

    #Test sample ids match
    assert torch.all(torch.eq(tensors[3], torch.Tensor([1,2])))


def test_multi_label_single_text_collate(multilabel_batch_sample, tokenizer):
    collated = single_text_collate_function(
        multilabel_batch_sample,
        'text',
        ['label1','label2','label3','label4'],
        dm.sample_id_col,
        tokenizer,
        label_encoder=None,
        prepare_target=True,
        prepare_sample_id=True
    )

    tensors = collect_tensors(collated)

    assert (len(tensors) == 4) #inputs tokens, input lengths, targets, ids
    assert (all(elem.shape[0] == len(multilabel_batch_sample) for elem in iter(tensors)))

    #Test tokens match
    assert torch.all(torch.eq(tensors[0][0], torch.Tensor([101,2034,102]))) #2034 = symbol for 'first'
    assert torch.all(torch.eq(tensors[0][1], torch.Tensor([101,2117,102])))

    #Test lengths match
    assert torch.all(torch.eq(tensors[1], torch.Tensor([3,3])))

    #Test labels match
    assert torch.all(torch.eq(tensors[2][0], torch.Tensor([0,0,0,0])))
    assert torch.all(torch.eq(tensors[2][1], torch.Tensor([0,1,0,1])))

    #Test sample ids match
    assert torch.all(torch.eq(tensors[3], torch.Tensor([1,2])))

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

def test_single_label_zero_window_text_collate(windowed_batch_sample, tokenizer, label_encoder):
    collated = windowed_text_collate_function(
        windowed_batch_sample,
        'text',
        'label',
        dm.sample_id_col,
        tokenizer,
        label_encoder=label_encoder,
        additional_text_cols = [],
        prev_sample_size=0,
        next_sample_size=0,
        concatenate=False,
        prepare_target=True,
        prepare_sample_id=True
    )

    tensors = collect_tensors(collated)

    assert (len(tensors) == 4) #inputs tokens, input lengths, targets, ids
    assert (all(elem.shape[0] == len(windowed_batch_sample) for elem in iter(tensors)))

    #Test tokens match
    assert torch.all(torch.eq(tensors[0][0], torch.Tensor([101,2959,102])))
    assert torch.all(torch.eq(tensors[0][1], torch.Tensor([101,3587,102])))

    #Test lengths match
    assert torch.all(torch.eq(tensors[1], torch.Tensor([3,3])))

    #Test labels match
    assert torch.all(torch.eq(tensors[2], torch.Tensor([0,1])))

    #Test sample ids match
    assert torch.all(torch.eq(tensors[3], torch.Tensor([1,2])))


    #why 20? 
    # 1 length tensor for each text column (9)
    # 1 token tensor for each text column (9)
    # 1 label tensor
    # 1 target id tensor

def test_single_label_concatenated_text_collate(windowed_batch_sample, tokenizer, label_encoder):
    collated = windowed_text_collate_function(
        windowed_batch_sample,
        'text',
        'label',
        dm.sample_id_col,
        tokenizer,
        label_encoder=label_encoder,
        additional_text_cols = [],
        prev_sample_size=3,
        next_sample_size=3,
        concatenate=True,
        prepare_target=True,
        prepare_sample_id=True
    )

    keys = collect_keys(collated)
    tensors = collect_tensors(collated)

    assert ('tokens' in keys)
    assert ('lengths' in keys)

    expected_tokens, expected_lengths = tokenizer.batch_encode([
        "fifth",
        "sixth",
        "first. second. third! fourth. fifth. sixth. seventh.",
        "second. third! fourth. fifth. sixth. seventh. eighth."
    ])

    #why 6? 
    # 1 length tensor for each text column (2)
    # 1 token tensor for each text column (2)
    # 1 label tensor
    # 1 target id tensor
    assert (len(tensors) == 6)
    assert (all(elem.shape[0] == len(windowed_batch_sample) for elem in iter(tensors)))

    #Test tokens match
    assert torch.all(torch.eq(tensors[0][0], torch.tensor([101,2959,102])))
    assert torch.all(torch.eq(tensors[0][1], torch.tensor([101,3587,102])))
    assert torch.all(torch.eq(tensors[1][0], expected_tokens[2]))
    assert torch.all(torch.eq(tensors[1][1], expected_tokens[3]))

    #Test lengths match
    assert torch.all(torch.eq(tensors[2], torch.Tensor([3,3])))
    assert torch.all(torch.eq(tensors[3], torch.Tensor([16,16])))

    #Test labels match
    assert torch.all(torch.eq(tensors[4], torch.Tensor([0,1])))

    #Test sample ids match
    assert torch.all(torch.eq(tensors[5], torch.Tensor([1,2])))

def test_single_label_no_concatenations_text_collate(windowed_batch_sample, tokenizer, label_encoder):
    collated = windowed_text_collate_function(
        windowed_batch_sample,
        'text',
        'label',
        dm.sample_id_col,
        tokenizer,
        label_encoder=label_encoder,
        additional_text_cols = [],
        prev_sample_size=3,
        next_sample_size=3,
        concatenate=False,
        prepare_target=True,
        prepare_sample_id=True
    )

    keys = collect_keys(collated)
    tensors = collect_tensors(collated)

    assert ('tokens' in keys)
    assert ('lengths' in keys)

    #why 16? 
    # 1 length tensor for each text column (7)
    # 1 token tensor for each text column (7)
    # 1 label tensor
    # 1 target id tensor
    assert (len(tensors) == 16)
    assert (all(elem.shape[0] == len(windowed_batch_sample) for elem in iter(tensors)))

    #Test tokens match
    assert torch.all(torch.eq(tensors[0][0], torch.tensor([101,2959,102]))) #text first sample
    assert torch.all(torch.eq(tensors[0][1], torch.tensor([101,3587,102]))) #text second sample
    assert torch.all(torch.eq(tensors[1][0], torch.tensor([101,2034,102]))) #prev_3 first sample
    assert torch.all(torch.eq(tensors[1][1], torch.tensor([101,2117,102]))) #prev_3 second sample
    assert torch.all(torch.eq(tensors[2][0], torch.tensor([101,2117,102,0]))) #prev_2 first sample with padding
    assert torch.all(torch.eq(tensors[2][1], torch.tensor([101,2353,999,102]))) #prev_2 second sample with !
    assert torch.all(torch.eq(tensors[3][0], torch.tensor([101,2353,999,102]))) #prev_1 first sample with !
    assert torch.all(torch.eq(tensors[3][1], torch.tensor([101,2959,102,0]))) #prev_1 second sample with padding
    assert torch.all(torch.eq(tensors[4][0], torch.tensor([101,3587,102]))) #next_1 first sample
    assert torch.all(torch.eq(tensors[4][1], torch.tensor([101,4369,102]))) #next_1 second sample
    assert torch.all(torch.eq(tensors[5][0], torch.tensor([101,4369,102]))) #next_2 first sample
    assert torch.all(torch.eq(tensors[5][1], torch.tensor([101,5066,102]))) #next_2 second sample
    assert torch.all(torch.eq(tensors[6][0], torch.tensor([101,5066,102]))) #next_3 first sample
    assert torch.all(torch.eq(tensors[6][1], torch.tensor([101,5964,102]))) #next_3 second sample

    #Test lengths match
    assert torch.all(torch.eq(tensors[7], torch.Tensor([3,3])))
    assert torch.all(torch.eq(tensors[8], torch.Tensor([3,3])))
    assert torch.all(torch.eq(tensors[9], torch.Tensor([4,4]))) #The padded one with !
    assert torch.all(torch.eq(tensors[10], torch.Tensor([4,4]))) #The other padded one
    assert torch.all(torch.eq(tensors[11], torch.Tensor([3,3])))
    assert torch.all(torch.eq(tensors[12], torch.Tensor([3,3])))
    assert torch.all(torch.eq(tensors[13], torch.Tensor([3,3])))

    #Test labels match
    assert torch.all(torch.eq(tensors[14], torch.Tensor([0,1])))

    #Test sample ids match
    assert torch.all(torch.eq(tensors[15], torch.Tensor([1,2])))

def test_multi_label_no_concatenations_text_collate(multilabel_windowed_batch_sample, tokenizer):
    collated = windowed_text_collate_function(
        multilabel_windowed_batch_sample,
        'text',
        ['label1', 'label2', 'label3', 'label4'],
        dm.sample_id_col,
        tokenizer,
        label_encoder=None,
        additional_text_cols = [],
        prev_sample_size=3,
        next_sample_size=3,
        concatenate=False,
        prepare_target=True,
        prepare_sample_id=True
    )

    keys = collect_keys(collated)
    tensors = collect_tensors(collated)

    assert ('tokens' in keys)
    assert ('lengths' in keys)

    #why 16? 
    # 1 length tensor for each text column (7)
    # 1 token tensor for each text column (7)
    # 1 label tensor
    # 1 target id tensor
    assert (len(tensors) == 16)
    assert (all(elem.shape[0] == len(multilabel_windowed_batch_sample) for elem in iter(tensors)))

    #Test tokens match
    assert torch.all(torch.eq(tensors[0][0], torch.tensor([101,2959,102]))) #text first sample
    assert torch.all(torch.eq(tensors[0][1], torch.tensor([101,3587,102]))) #text second sample
    assert torch.all(torch.eq(tensors[1][0], torch.tensor([101,2034,102]))) #prev_3 first sample
    assert torch.all(torch.eq(tensors[1][1], torch.tensor([101,2117,102]))) #prev_3 second sample
    assert torch.all(torch.eq(tensors[2][0], torch.tensor([101,2117,102,0]))) #prev_2 first sample with padding
    assert torch.all(torch.eq(tensors[2][1], torch.tensor([101,2353,999,102]))) #prev_2 second sample with !
    assert torch.all(torch.eq(tensors[3][0], torch.tensor([101,2353,999,102]))) #prev_1 first sample with !
    assert torch.all(torch.eq(tensors[3][1], torch.tensor([101,2959,102,0]))) #prev_1 second sample with padding
    assert torch.all(torch.eq(tensors[4][0], torch.tensor([101,3587,102]))) #next_1 first sample
    assert torch.all(torch.eq(tensors[4][1], torch.tensor([101,4369,102]))) #next_1 second sample
    assert torch.all(torch.eq(tensors[5][0], torch.tensor([101,4369,102]))) #next_2 first sample
    assert torch.all(torch.eq(tensors[5][1], torch.tensor([101,5066,102]))) #next_2 second sample
    assert torch.all(torch.eq(tensors[6][0], torch.tensor([101,5066,102]))) #next_3 first sample
    assert torch.all(torch.eq(tensors[6][1], torch.tensor([101,5964,102]))) #next_3 second sample

    #Test lengths match
    assert torch.all(torch.eq(tensors[7], torch.Tensor([3,3])))
    assert torch.all(torch.eq(tensors[8], torch.Tensor([3,3])))
    assert torch.all(torch.eq(tensors[9], torch.Tensor([4,4]))) #The padded one with !
    assert torch.all(torch.eq(tensors[10], torch.Tensor([4,4]))) #The other padded one
    assert torch.all(torch.eq(tensors[11], torch.Tensor([3,3])))
    assert torch.all(torch.eq(tensors[12], torch.Tensor([3,3])))
    assert torch.all(torch.eq(tensors[13], torch.Tensor([3,3])))

    #Test labels match
    assert torch.all(torch.eq(tensors[14][0], torch.Tensor([0,0,0,0])))
    assert torch.all(torch.eq(tensors[14][1], torch.Tensor([0,1,0,1])))

    #Test sample ids match
    assert torch.all(torch.eq(tensors[15], torch.Tensor([1,2])))

def test_sampled_window_text_collate(sample_windowed_batch_sample, tokenizer, label_encoder):

    num_iterations = 10

    def tensor_equals_one_of_many(a, b_list):
        for b in b_list:
            if torch.equal(a,b): return True
        return False

    accepted_previous_first_sample = [
        torch.tensor([101,2034,102]), torch.tensor([101,2117,102]), torch.tensor([101,2353,102])
    ]

    accepted_previous_second_sample = [
        torch.tensor([101,2117,102]), torch.tensor([101,2353,102]), torch.tensor([101,2959,102])
    ]

    accepted_next_first_sample = [
        torch.tensor([101,3587,102]), torch.tensor([101,4369,102]), torch.tensor([101,5066,102])
    ]

    accepted_next_second_sample = [
        torch.tensor([101,4369,102]), torch.tensor([101,5066,102]), torch.tensor([101,5964,102])
    ]

    for _ in range(num_iterations):
        collated = windowed_text_collate_function(
            sample_windowed_batch_sample,
            'text',
            'label',
            dm.sample_id_col,
            tokenizer,
            label_encoder=label_encoder,
            additional_text_cols = [],
            prev_sample_size=1,
            next_sample_size=1,
            concatenate=False,
            prepare_target=True,
            prepare_sample_id=True
        )

        tensors = collect_tensors(collated)

        #Test tokens match
        assert torch.all(torch.eq(tensors[0][0], torch.tensor([101,2959,102])))
        assert torch.all(torch.eq(tensors[0][1], torch.tensor([101,3587,102])))
        assert tensor_equals_one_of_many(tensors[1][0], accepted_previous_first_sample)
        assert tensor_equals_one_of_many(tensors[1][1], accepted_previous_second_sample)
        assert tensor_equals_one_of_many(tensors[2][0], accepted_next_first_sample)
        assert tensor_equals_one_of_many(tensors[2][1], accepted_next_second_sample)

def test_evaluation_sampled_window_text_collate(sample_windowed_batch_sample, tokenizer, label_encoder):

    num_iterations = 10

    for _ in range(num_iterations):
        collated = windowed_text_collate_function(
            sample_windowed_batch_sample,
            'text',
            'label',
            dm.sample_id_col,
            tokenizer,
            label_encoder=label_encoder,
            additional_text_cols = [],
            prev_sample_size=2,
            next_sample_size=2,
            concatenate=False,
            prepare_target=False,
            prepare_sample_id=True
        )

        tensors = collect_tensors(collated)

        assert (len(tensors) == 11)
        assert (all(elem.shape[0] == len(sample_windowed_batch_sample) for elem in iter(tensors)))

        #Test tokens match
        assert torch.all(torch.eq(tensors[0][0], torch.tensor([101,2959,102]))) #sample 1 text                
        assert torch.all(torch.eq(tensors[0][1], torch.tensor([101,3587,102]))) #sample 2 text  
        assert torch.all(torch.eq(tensors[1][0], torch.tensor([101,2117,102]))) #prev 2 sample 1 "second"
        assert torch.all(torch.eq(tensors[1][1], torch.tensor([101,2353,102]))) #prev 2 sample 2 "third"
        assert torch.all(torch.eq(tensors[2][0], torch.tensor([101,2353,102]))) #prev 1 sample 1 "third"
        assert torch.all(torch.eq(tensors[2][1], torch.tensor([101,2959,102]))) #prev 1 sample 2 "fourth"
        assert torch.all(torch.eq(tensors[3][0], torch.tensor([101,3587,102]))) #next 1 sample 1 "fifth"
        assert torch.all(torch.eq(tensors[3][1], torch.tensor([101,4369,102]))) #next 1 sample 2 "sixth"
        assert torch.all(torch.eq(tensors[4][0], torch.tensor([101,4369,102]))) #next 2 sample 1 "sixth"
        assert torch.all(torch.eq(tensors[4][1], torch.tensor([101,5066,102]))) #next 2 sample 2 "seventh"

        #Test lengths match
        assert torch.all(torch.eq(tensors[5], torch.Tensor([3,3])))
        assert torch.all(torch.eq(tensors[6], torch.Tensor([3,3])))
        assert torch.all(torch.eq(tensors[7], torch.Tensor([3,3])))
        assert torch.all(torch.eq(tensors[8], torch.Tensor([3,3])))
        assert torch.all(torch.eq(tensors[9], torch.Tensor([3,3])))

        #Test labels match, but there are no labels!

        #Test sample ids match
        assert torch.all(torch.eq(tensors[10], torch.Tensor([1,2])))

def test_evaluation_concatenated_sampled_window_text_collate(sample_windowed_batch_sample, tokenizer, label_encoder):

    num_iterations = 10

    expected_tokens, _ = tokenizer.batch_encode([
        "fourth",
        "fifth",
        "second. third. fourth. fifth. sixth.",
        "third. fourth. fifth. sixth. seventh."
    ])

    for _ in range(num_iterations):
        collated = windowed_text_collate_function(
            sample_windowed_batch_sample,
            'text',
            'label',
            dm.sample_id_col,
            tokenizer,
            label_encoder=label_encoder,
            additional_text_cols = [],
            prev_sample_size=2,
            next_sample_size=2,
            concatenate=True,
            prepare_target=False,
            prepare_sample_id=True
        )

        tensors = collect_tensors(collated)

        assert (len(tensors) == 5)
        assert (all(elem.shape[0] == len(sample_windowed_batch_sample) for elem in iter(tensors)))

        #Test tokens match
        assert torch.all(torch.eq(tensors[0][0], torch.tensor([101,2959,102]))) #sample text 1
        assert torch.all(torch.eq(tensors[0][1], torch.tensor([101,3587,102]))) #sample text 2
        assert torch.all(torch.eq(tensors[1][0], expected_tokens[2])) #concatenated sample 1
        assert torch.all(torch.eq(tensors[1][1], expected_tokens[3])) #concatenated sample 2

        #Test lengths match
        assert torch.all(torch.eq(tensors[2], torch.Tensor([3,3])))
        assert torch.all(torch.eq(tensors[3], torch.Tensor([12,12])))

        #Test labels match, but there are no labels!

        #Test sample ids match
        assert torch.all(torch.eq(tensors[4], torch.Tensor([1,2])))