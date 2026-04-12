"""Tests for sheepy.nlp.label_encoder — migrated from original test suite."""
import pandas as pd
import pytest
import torch

from sheepy.nlp.label_encoder import LabelEncoder


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "label": ["cat", "dog", "bird", "cat", "dog"],
        "text": ["meow", "woof", "tweet", "purr", "bark"],
    })


class TestLabelEncoderInit:
    def test_vocab_order(self):
        enc = LabelEncoder(["a", "b", "c"])
        assert enc.vocab == ["a", "b", "c"]
        assert enc.vocab_size == 3

    def test_with_unknown_label(self):
        enc = LabelEncoder(["a", "b", "UNK"], unknown_label="UNK")
        assert enc.unknown_label == "UNK"
        assert enc.vocab_size == 3


class TestEncode:
    def test_known_label(self):
        enc = LabelEncoder(["spam", "ham"])
        result = enc.encode("ham")
        assert result.item() == 1

    def test_unknown_raises(self):
        enc = LabelEncoder(["a", "b"])
        with pytest.raises(ValueError):
            enc.encode("c")

    def test_unknown_falls_back(self):
        enc = LabelEncoder(["a", "b", "UNK"], unknown_label="UNK")
        result = enc.encode("xyz")
        assert enc.decode(result) == "UNK"


class TestBatchEncode:
    def test_basic(self):
        enc = LabelEncoder(["x", "y", "z"])
        result = enc.batch_encode(["z", "x", "y"])
        assert result.tolist() == [2, 0, 1]


class TestDecode:
    def test_roundtrip(self):
        enc = LabelEncoder(["alpha", "beta"])
        for label in ["alpha", "beta"]:
            assert enc.decode(enc.encode(label)) == label

    def test_batch_roundtrip(self):
        enc = LabelEncoder(["a", "b", "c"])
        labels = ["c", "a", "b", "a"]
        encoded = enc.batch_encode(labels)
        decoded = enc.batch_decode(encoded)
        assert decoded == labels


class TestFromDataframe:
    def test_basic(self, sample_df):
        enc = LabelEncoder.initializeFromDataframe(sample_df, "label")
        assert enc.vocab_size == 3
        assert set(enc.vocab) == {"cat", "dog", "bird"}

    def test_with_reserved_labels(self, sample_df):
        enc = LabelEncoder.initializeFromDataframe(
            sample_df, "label", reserved_labels=["fish"]
        )
        assert enc.vocab[0] == "fish"
        assert enc.vocab_size == 4

    def test_with_unknown(self, sample_df):
        enc = LabelEncoder.initializeFromDataframe(
            sample_df, "label", unknown_label="OTHER"
        )
        assert "OTHER" in enc.vocab


class TestMultilabel:
    def test_from_multilabel_dataframe(self):
        df = pd.DataFrame({"toxic": [0, 1, 0], "insult": [1, 0, 1]})
        enc = LabelEncoder.initializeFromMultilabelDataframe(df, ["toxic", "insult"])
        assert enc.multilabel is True
        assert enc.vocab == ["toxic", "insult"]

    def test_batch_encode_multilabel(self):
        enc = LabelEncoder(["a", "b"], multilabel=True)
        sample = {"a": ["1", "0", "1"], "b": ["0", "1", "1"]}
        result = enc.batch_encode_multilabel(sample)
        assert result.shape == (3, 2)
        assert result.dtype == torch.float32

    def test_non_binary_raises(self):
        df = pd.DataFrame({"x": [0, 1, 2]})
        with pytest.raises(ValueError, match="must be binary"):
            LabelEncoder.initializeFromMultilabelDataframe(df, ["x"])
