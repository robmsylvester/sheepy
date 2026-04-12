"""Tests for sheepy.common.collate."""
import pytest

from sheepy.common.collate import (
    _concatenate_text_samples,
    _get_sample_list,
    collate_tensors,
    round_size,
)


class TestCollateTensors:
    def test_basic(self):
        batch = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        result = collate_tensors(batch)
        assert result == {"a": [1, 2], "b": ["x", "y"]}

    def test_single_item(self):
        result = collate_tensors([{"key": "val"}])
        assert result == {"key": ["val"]}


class TestGetSampleList:
    def test_default_size(self):
        result = _get_sample_list(["a", "b", "c"])
        assert len(result) == 3

    def test_smaller_sample(self):
        result = _get_sample_list(["a", "b", "c", "d"], sample_size=2)
        assert len(result) == 2

    def test_too_large_raises(self):
        with pytest.raises(ValueError):
            _get_sample_list(["a", "b"], sample_size=5)


class TestConcatenateTextSamples:
    def test_basic(self):
        sample = {"col1": ["Hello", "Hi"], "col2": ["world", "there"]}
        result = _concatenate_text_samples(sample, ["col1", "col2"])
        assert result[0] == "Hello. world."
        assert result[1] == "Hi. there."

    def test_with_punctuation(self):
        sample = {"a": ["Hello!", "What?"], "b": ["next", "part"]}
        result = _concatenate_text_samples(sample, ["a", "b"])
        assert result[0] == "Hello! next."
        assert result[1] == "What? part."


class TestRoundSize:
    def test_powers_of_two(self):
        assert round_size(1) == 2
        assert round_size(2) == 2
        assert round_size(3) == 4
        assert round_size(5) == 8
        assert round_size(100) == 128
