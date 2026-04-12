"""DataFrame operations for loading, splitting, and resampling classification datasets."""
from __future__ import annotations

import os
import random
from copy import deepcopy
from typing import Any

import pandas as pd

from sheepy.common.logger import get_logger

logger = get_logger(__name__)

SOURCE_FILE = "source_file"


def column_exists(col_name: str, df: pd.DataFrame) -> bool:
    """Return True if *col_name* is present in *df*."""
    return col_name in df.columns


def read_csv(path: str, filter_cols: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV with optional column filtering."""
    df = pd.read_csv(path)
    if filter_cols is not None:
        df = df.filter(filter_cols)
    df[SOURCE_FILE] = os.path.basename(path)
    return df.astype(str)


def read_csv_text_classifier(
    path: str,
    encoding: str = "utf-8",
    delimiter: str = ",",
    evaluate: bool = False,
    label_cols: str | list[str] = "label",
    text_col: str = "text",
    additional_cols: list[str] | None = None,
    names: list[str] | None = None,
) -> pd.DataFrame:
    """Read a CSV for text classification, returning only necessary columns."""
    if names is None:
        df = pd.read_csv(path, encoding=encoding, delimiter=delimiter)
    else:
        df = pd.read_csv(path, encoding=encoding, names=names, delimiter=delimiter)

    labels = deepcopy(label_cols)
    if isinstance(labels, str):
        labels = [labels]

    if not evaluate:
        for col in labels:
            if not column_exists(col, df):
                raise ValueError(f"'{col}' not a valid label column in {path}")

    if text_col is None or not column_exists(text_col, df):
        raise ValueError(
            f"'{text_col}' is not a valid text column. Must exist in the dataframe."
        )

    if evaluate:
        return_df = df.copy()
        return_df[SOURCE_FILE] = os.path.basename(path)
    else:
        return_columns = labels.copy()
        if text_col not in return_columns:
            return_columns.append(text_col)
        return_df = df[return_columns].copy()

        if isinstance(additional_cols, list):
            for col in additional_cols:
                if column_exists(col, df) and not column_exists(col, return_df):
                    return_df[col] = df[col]

    return return_df.astype(str)


def add_relative_position(document: pd.DataFrame) -> pd.DataFrame:
    """Add a ``relative_position`` column that maps row index to [-1, 1]."""
    n = document.shape[0]
    document = document.copy()
    document["relative_position"] = ((pd.RangeIndex(n) / n) * 2) - 1
    return document


def split_dataframes(
    dataframes: list[pd.DataFrame],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float | None = None,
    shuffle: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a list of DataFrames into train / val / test according to ratios."""
    if len(dataframes) == 1:
        return split_dataframe(dataframes[0], train_ratio, validation_ratio, shuffle=shuffle)

    first_val = round(len(dataframes) * train_ratio)
    first_test = round(len(dataframes) * (train_ratio + validation_ratio))

    if first_val == 0 or (first_val == first_test and validation_ratio > 0):
        combined = pd.concat(dataframes)
        return split_dataframe(combined, train_ratio, validation_ratio, shuffle=shuffle)

    if shuffle:
        random.shuffle(dataframes)

    train = pd.concat(dataframes[:first_val])
    val_dfs = dataframes[first_val:first_test]
    test_dfs = dataframes[first_test:]
    val = pd.concat(val_dfs) if val_dfs else pd.DataFrame(columns=train.columns.tolist())
    test = pd.concat(test_dfs) if test_dfs else pd.DataFrame(columns=train.columns.tolist())
    return train, val, test


def split_dataframe(
    df: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float | None = None,
    shuffle: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a single DataFrame into train / val / test."""
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    n = len(df)
    first_val = round(n * train_ratio)
    first_test = round(n * (train_ratio + validation_ratio))
    return df.iloc[:first_val], df.iloc[first_val:first_test], df.iloc[first_test:]


def write_csv_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Write *df* to CSV, splitting by ``SOURCE_FILE`` when *output_path* is a directory."""
    if os.path.isdir(output_path):
        for source_file, results in df.groupby(SOURCE_FILE, as_index=False):
            results = results.drop(SOURCE_FILE, axis=1)
            results.to_csv(os.path.join(output_path, source_file), index=False)
    elif os.path.isfile(output_path) and output_path.endswith(".csv"):
        df.to_csv(output_path, index=False)


def write_json_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Write *df* as JSON records, splitting by ``SOURCE_FILE`` when *output_path* is a directory."""
    if os.path.isdir(output_path):
        for source_file, results in df.groupby(SOURCE_FILE, as_index=False):
            results = results.drop(SOURCE_FILE, axis=1)
            results.to_json(os.path.join(output_path, source_file), orient="records")
    elif os.path.isfile(output_path) and output_path.endswith(".json"):
        df.to_json(output_path, orient="records")


def map_labels(df: pd.DataFrame, label_col: str, label_map: dict) -> pd.DataFrame:
    """Apply a label mapping to a column."""
    df = df.copy()
    df[label_col] = df[label_col].map(label_map)
    return df


def resample_positives(
    df: pd.DataFrame,
    resample_rate: int,
    label_col: str,
    pos_label_val: Any,
    reshuffle: bool = True,
) -> pd.DataFrame:
    """Upsample rows matching a positive label value."""
    if not isinstance(resample_rate, int) or resample_rate < 1:
        raise ValueError("Resample rate must be a positive integer.")
    if resample_rate > 1:
        positives = df[df[label_col] == pos_label_val]
        copies = [positives] * (resample_rate - 1)
        df = pd.concat([df, *copies], ignore_index=True)
    if reshuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    return df


def resample_multilabel_positives(
    df: pd.DataFrame,
    resample_rate_dict: dict[str, int],
    pos_label_val: Any = "1",
    reshuffle: bool = True,
) -> pd.DataFrame:
    """Upsample rows for each label column according to per-label rates."""
    copies = []
    for label_col, rate in resample_rate_dict.items():
        if not isinstance(rate, int) or rate < 1:
            raise ValueError(f"Rate must be a positive integer; got {rate} for '{label_col}'.")
        if label_col not in df.columns:
            raise ValueError(f"'{label_col}' not in dataframe columns.")
        if rate > 1:
            positives = df[df[label_col] == pos_label_val]
            copies.extend([positives] * (rate - 1))

    if copies:
        df = pd.concat([df, *copies], ignore_index=True)
    if reshuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    return df
