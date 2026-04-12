"""Tests for sheepy.common.df_ops."""
import os

import pandas as pd
import pytest

from sheepy.common.df_ops import (
    column_exists,
    map_labels,
    read_csv,
    read_csv_text_classifier,
    resample_positives,
    split_dataframe,
    split_dataframes,
)

RESOURCES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")


class TestColumnExists:
    def test_exists(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert column_exists("a", df) is True
        assert column_exists("c", df) is False


class TestReadCSV:
    def test_read_basic(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("text,label\nhello,spam\nworld,ham\n")
        df = read_csv(str(f))
        assert len(df) == 2
        assert "source_file" in df.columns

    def test_filter_cols(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("a,b,c\n1,2,3\n")
        df = read_csv(str(f), filter_cols=["a", "c"])
        assert "b" not in df.columns


class TestReadCSVTextClassifier:
    def test_basic(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("text,label\nhello,spam\nworld,ham\n")
        df = read_csv_text_classifier(str(f))
        assert "text" in df.columns
        assert "label" in df.columns

    def test_missing_label_raises(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("text,other\nhello,1\n")
        with pytest.raises(ValueError):
            read_csv_text_classifier(str(f), label_cols="label")

    def test_missing_text_raises(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("label,other\nspam,1\n")
        with pytest.raises(ValueError):
            read_csv_text_classifier(str(f), text_col="text")

    def test_eval_mode(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("text,label,extra\nhello,spam,1\n")
        df = read_csv_text_classifier(str(f), evaluate=True)
        assert "extra" in df.columns


class TestSplitDataframe:
    def test_basic_split(self):
        df = pd.DataFrame({"x": range(100)})
        train, val, test = split_dataframe(df, 0.8, 0.1)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_split_dataframes_single(self):
        dfs = [pd.DataFrame({"x": range(50)})]
        train, val, test = split_dataframes(dfs, 0.6, 0.2)
        assert len(train) + len(val) + len(test) == 50


class TestMapLabels:
    def test_basic(self):
        df = pd.DataFrame({"label": ["a", "b", "a"]})
        result = map_labels(df, "label", {"a": "alpha", "b": "beta"})
        assert result["label"].tolist() == ["alpha", "beta", "alpha"]


class TestResamplePositives:
    def test_rate_2(self):
        df = pd.DataFrame({"label": ["a"] * 80 + ["b"] * 20, "text": ["x"] * 100})
        result = resample_positives(df, 2, "label", "b", reshuffle=False)
        assert len(result) == 120

    def test_invalid(self):
        df = pd.DataFrame({"label": ["a"]})
        with pytest.raises(ValueError):
            resample_positives(df, 0, "label", "a")
