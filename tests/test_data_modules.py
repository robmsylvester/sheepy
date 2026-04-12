"""Tests for sheepy data modules."""
import os

import pandas as pd
import pytest

from sheepy.common.df_ops import (
    read_csv_text_classifier,
    resample_multilabel_positives,
    resample_positives,
    split_dataframe,
)
from sheepy.nlp.label_encoder import LabelEncoder


RESOURCES = os.path.join(os.path.dirname(__file__), "resources")


class TestReadCSV:
    def test_read_valid_csv(self):
        path = os.path.join(RESOURCES, "dummy_dataset.csv")
        if not os.path.exists(path):
            pytest.skip("dummy_dataset.csv not found")
        df = read_csv_text_classifier(path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_missing_text_col_raises(self, tmp_path):
        f = tmp_path / "bad.csv"
        # Include a valid label column so validation reaches the text-column check.
        f.write_text("label,col_b\nham,hello\n")
        with pytest.raises(ValueError, match="not a valid text column"):
            read_csv_text_classifier(str(f), text_col="nonexistent", label_cols="label")


class TestSplitDataframe:
    def test_split_ratios(self):
        df = pd.DataFrame({"x": range(100)})
        train, val, test = split_dataframe(df, 0.7, 0.15)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_split_with_shuffle(self):
        df = pd.DataFrame({"x": range(100)})
        train1, _, _ = split_dataframe(df, 0.7, 0.15, shuffle=True)
        train2, _, _ = split_dataframe(df, 0.7, 0.15, shuffle=True)
        # With high probability these will differ
        assert len(train1) == len(train2) == 70


class TestResample:
    def test_resample_positives(self):
        df = pd.DataFrame({"label": ["a"] * 90 + ["b"] * 10, "text": ["x"] * 100})
        result = resample_positives(df, 3, "label", "b", reshuffle=False)
        assert len(result) == 120  # 100 + 10*2

    def test_resample_rate_1_noop(self):
        df = pd.DataFrame({"label": ["a"] * 50 + ["b"] * 50, "text": ["x"] * 100})
        result = resample_positives(df, 1, "label", "b")
        assert len(result) == 100

    def test_resample_invalid_rate(self):
        df = pd.DataFrame({"label": ["a"], "text": ["x"]})
        with pytest.raises(ValueError):
            resample_positives(df, -1, "label", "a")


class TestResampleMultilabel:
    def test_basic(self):
        df = pd.DataFrame({
            "text": ["t1", "t2", "t3", "t4"],
            "A": ["0", "1", "0", "1"],
            "B": ["1", "0", "0", "0"],
        })
        result = resample_multilabel_positives(df, {"A": 2, "B": 3}, reshuffle=False)
        # Original 4 + 2 copies of A=1 (1 extra) + 1 copy of B=1 (2 extra)
        assert len(result) == 4 + 2 + 2

    def test_invalid_column_raises(self):
        df = pd.DataFrame({"text": ["a"]})
        with pytest.raises(ValueError, match="not in dataframe"):
            resample_multilabel_positives(df, {"missing": 2})


class TestLabelEncoder:
    def test_encode_decode_roundtrip(self):
        enc = LabelEncoder(["cat", "dog", "bird"])
        encoded = enc.encode("dog")
        assert enc.decode(encoded) == "dog"

    def test_batch_encode(self):
        enc = LabelEncoder(["a", "b", "c"])
        result = enc.batch_encode(["a", "c", "b"])
        assert result.tolist() == [0, 2, 1]

    def test_unknown_label_raises(self):
        enc = LabelEncoder(["a", "b"])
        with pytest.raises(ValueError, match="Unknown label"):
            enc.encode("z")

    def test_unknown_label_fallback(self):
        enc = LabelEncoder(["a", "b", "UNK"], unknown_label="UNK")
        result = enc.encode("z")
        assert enc.decode(result) == "UNK"

    def test_from_dataframe(self):
        df = pd.DataFrame({"label": ["spam", "ham", "spam", "ham"]})
        enc = LabelEncoder.initializeFromDataframe(df, "label")
        assert set(enc.vocab) == {"spam", "ham"}
        assert enc.vocab_size == 2

    def test_from_multilabel_dataframe(self):
        df = pd.DataFrame({"toxic": [0, 1, 0], "obscene": [1, 1, 0]})
        enc = LabelEncoder.initializeFromMultilabelDataframe(df, ["toxic", "obscene"])
        assert enc.vocab == ["toxic", "obscene"]
        assert enc.multilabel is True
