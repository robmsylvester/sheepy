import pytest
import torch
from pandas import DataFrame

from sheepy.nlp.label_encoder import LabelEncoder


@pytest.fixture
def binary_label_encoder() -> LabelEncoder:
    d = {
        "num_purchases": [5, 10, 15, 20, 25, 30, 35],
        "label": ["apple", "banana", "orange", "apple", "apple", "strawberry", "pear"],
    }
    df = DataFrame(data=d)
    encoder = LabelEncoder.initializeFromDataframe(df, label_col="label")
    return encoder


@pytest.fixture
def binary_label_encoder_unknown_exists() -> LabelEncoder:
    d = {
        "num_purchases": [5, 10, 15, 20, 25, 30, 35],
        "label": ["apple", "banana", "orange", "apple", "apple", "strawberry", "pear"],
    }
    df = DataFrame(data=d)
    encoder = LabelEncoder.initializeFromDataframe(df, label_col="label", unknown_label="apple")
    return encoder


@pytest.fixture
def binary_label_encoder_unknown_does_not_exist() -> LabelEncoder:
    d = {
        "num_purchases": [5, 10, 15, 20, 25, 30, 35],
        "label": ["apple", "banana", "orange", "apple", "apple", "strawberry", "pear"],
    }
    df = DataFrame(data=d)
    encoder = LabelEncoder.initializeFromDataframe(df, label_col="label", unknown_label="unknown")
    return encoder


@pytest.fixture
def binary_label_encoder_reserved_vocab() -> LabelEncoder:
    d = {
        "num_purchases": [5, 10, 15, 20, 25, 30, 35],
        "label": ["apple", "banana", "orange", "apple", "apple", "strawberry", "pear"],
    }
    df = DataFrame(data=d)
    encoder = LabelEncoder.initializeFromDataframe(
        df, label_col="label", reserved_labels=["reserved1", "reserved2"], unknown_label="unknown"
    )
    return encoder


# Test the single label factory functions
def test_binary_label_encoder_vocab(binary_label_encoder):
    assert binary_label_encoder.vocab == ["apple", "banana", "orange", "strawberry", "pear"]
    assert binary_label_encoder.vocab_size == 5


def test_binary_label_encoder_unknown_vocab_exists(binary_label_encoder_unknown_exists):
    assert binary_label_encoder_unknown_exists.vocab == [
        "apple",
        "banana",
        "orange",
        "strawberry",
        "pear",
    ]
    assert binary_label_encoder_unknown_exists.vocab_size == 5


def test_binary_label_encoder_unknown_vocab_does_not_exist(
    binary_label_encoder_unknown_does_not_exist,
):
    assert binary_label_encoder_unknown_does_not_exist.vocab == [
        "apple",
        "banana",
        "orange",
        "strawberry",
        "pear",
        "unknown",
    ]
    assert binary_label_encoder_unknown_does_not_exist.vocab_size == 6


def test_binary_label_encoder_reserved_vocab(binary_label_encoder_reserved_vocab):
    assert binary_label_encoder_reserved_vocab.vocab == [
        "reserved1",
        "reserved2",
        "apple",
        "banana",
        "orange",
        "strawberry",
        "pear",
        "unknown",
    ]
    assert binary_label_encoder_reserved_vocab.vocab_size == 8


# Test the single label encoder functions
def test_binary_label_encoder_batch_encodes(binary_label_encoder):
    test_strs = ["banana", "banana", "pear", "apple"]

    expected = torch.tensor([1, 1, 4, 0], dtype=torch.long)
    out = binary_label_encoder.batch_encode(test_strs)

    assert torch.equal(out, expected)


def test_binary_label_encoder_batch_encodes_with_unknown_exists(
    binary_label_encoder_unknown_exists,
):
    test_strs = ["banana", "banana", "pear", "this_is_unknown", "apple"]

    expected = torch.tensor([1, 1, 4, 0, 0], dtype=torch.long)
    out = binary_label_encoder_unknown_exists.batch_encode(test_strs)

    assert torch.equal(out, expected)


def test_binary_label_encoder_batch_encodes_with_unknown_does_not_exist(
    binary_label_encoder_unknown_does_not_exist,
):
    test_strs = ["banana", "banana", "pear", "this_is_unknown", "apple"]

    expected = torch.tensor([1, 1, 4, 5, 0], dtype=torch.long)
    out = binary_label_encoder_unknown_does_not_exist.batch_encode(test_strs)

    assert torch.equal(out, expected)


# Test the single label decoder functions
def test_binary_label_encoder_batch_decodes(binary_label_encoder):
    expected = ["banana", "banana", "pear", "apple"]
    test_encoded = torch.tensor([1, 1, 4, 0], dtype=torch.long)
    out = binary_label_encoder.batch_decode(test_encoded)

    assert expected == out


def test_binary_label_encoder_batch_decodes_with_unknown_does_not_exist(
    binary_label_encoder_unknown_does_not_exist,
):
    expected = ["banana", "banana", "pear", "unknown", "apple"]
    test_encoded = torch.tensor([1, 1, 4, 5, 0], dtype=torch.long)
    out = binary_label_encoder_unknown_does_not_exist.batch_decode(test_encoded)

    assert expected == out


# TODO - test some encoder->decoder->recover original
