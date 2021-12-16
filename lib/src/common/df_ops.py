import json
import os
import pandas as pd
import random
import json
from typing import Any, Dict, List, Union, Tuple
from copy import deepcopy

# TODO - move this global variable
SOURCE_FILE = 'source_file'

#TODO - docstrings

def column_exists(col_name: str, my_df: pd.DataFrame):
    """Helper function for readability to check if a column exists in a dataframe"""
    return True if col_name in my_df.columns else False

def read_csv(path: str,
             filter_cols: List[str]=None) -> pd.DataFrame:
    """
    Reads a full comma-separated value file and returns all columns. Some safety checks

    Arguments:
        path (str): path to a csv file.
        filter_cols (list(str)): list of columns to grab. If None, all columns will be grabbed
    Returns:
        pd.DataFrame: A pandas dataframe.
    """
    df = pd.read_csv(path)
    if filter_cols is not None:
        df = df.filter(filter_cols)
    df[SOURCE_FILE] = os.path.basename(path)

    return df.astype(str)

def read_csv_text_classifier(path: str,
                             encoding: str="utf-8",
                             delimiter: str=",",
                             evaluate: bool = False,
                             label_cols: Union[str, List[str]] = "label",
                             text_col: str = "text",
                             additional_cols: List[str] = None,
                             names: List[str] = None) -> pd.DataFrame:
    """
    Reads a full comma-separated value file that stores, at the minimum, a text column as well as a label column.

    Arguments:
        path (str): path to a csv file.
        encoding (str): type of encoding of csv
        delimiter (str): separation char/str for dataset. usually comma
        evaluate (bool): prepares data for the evaluation mode instead of the train/val/test mode
        label_cols (list or str): name of label column(s) in csv.
        text_col (str): name of text column in csv if it exists.
        additional_cols (list(str)): any additional columns to grab.
        names (str): If the header is 0, then we need to pass column names to the csv

    Returns:
        pd.DataFrame: A pandas dataframe containing just the text column and label column, as well as any additional columns passed
        In this dataframe, we return every value as a string object, regardless of original column type.

    Raises:
        ValueError:
        1. If the text column does not exist in the dataframe, OR
        2. If any of the label column does not exist in the dataframe and we're not in evaluation mode
    """
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
                raise ValueError(
                    "{} not a valid label column in dataset csv at {}".format(col, path))

    if text_col is None or not column_exists(text_col, df):
        raise ValueError(
            f"{text_col} is not a valid text column. Your text column must be a string that exists in the dataframe")

    if evaluate:
        return_df = df  # preserving all columns because in eval mode we have to output all columns
        # TODO - buggy, pushes nan values to all other columns
        return_df[SOURCE_FILE] = os.path.basename(path)
    else:
        return_columns = labels
        if text_col not in return_columns:
            return_columns.append(text_col)
        return_df = df[return_columns]

        if isinstance(additional_cols, list):
            for col in additional_cols:
                if column_exists(col, df) and not column_exists(col, return_df):
                    return_df[col] = df[col]

    return return_df.astype(str)


def window_text(df: pd.DataFrame, text_col: str = "text", n_prev_text_samples: int = 0, n_next_text_samples: int = 0):
    """Given a dataframe with a targeted text column, generate columns n_prev_text_samples to the left and 
        n_next_text_samples to the right that represent shifts of the targeted column, thereby allowing you
        to have in a given row a window of conversation context to the left and right.

    Args:
        df (pd.DataFrame): the dataframe to modify (not in place)
        text_col (str, optional): [description]. The target column to window Defaults to "text".
        n_prev_text_samples (int, optional): The number of previous text columns to window. Defaults to 0.
        n_next_text_samples (int, optional): The number of next text columns to window. Defaults to 0.

    Raises:
        ValueError: The text column is not in the dataframe

    Returns:
        [type]: The modified dataframe with the new columns, with names "{}_prev_{}".format(text_col, i) and "{}_next_{}".format(text_col, i)
    """
    if text_col not in df.columns:
        raise ValueError(
            "Cannot window text column {} that does not exist in the dataframe. Check config JSON and args to data modules".format(
                text_col))
    for i in range(1, n_prev_text_samples + 1):
        col_name = "{}_prev_{}".format(text_col, i)
        df[col_name] = df[text_col].shift(i).fillna("")
    for i in range(1, n_next_text_samples + 1):
        col_name = "{}_next_{}".format(text_col, i)
        df[col_name] = df[text_col].shift(-i).fillna("")
    return df


def add_relative_position(document: pd.DataFrame, index_col=None) -> pd.DataFrame:
    """
    Given a dataframe and a possibly-specified index colummn, otherwise using the default,
    create a new column named 'relative_position' that stores a [-1, 1] uniform distribution holding
    the row index. In other words, if your dataframe has 100 rows, the 50th row would have a relative_position
    of 0.0, and the 25th row would have a relative_position of -0.5.
    """
    document['temp_dummy'] = range(document.shape[0])
    document['relative_position'] = (
        (document['temp_dummy'] / document.shape[0]) * 2) - 1
    document.drop(columns=['temp_dummy'], inplace=True)
    return document


def split_dataframes(dataframes: List[pd.DataFrame], train_ratio: float, validation_ratio: float,
                     test_ratio: float = None, shuffle: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataframes to train, valid and test dfs according to the provided ratios.

    :param dataframes: list of dataframes to split
    :param train_ratio: float
    :param validation_ratio: float
    :param test_ratio: float
    :param shuffle: bool
    :return: tuple of 3 dataframes
    """

    if len(dataframes) == 1:
        return split_dataframe(dataframes[0], train_ratio, validation_ratio, shuffle)

    first_validation_index = round(len(dataframes) * train_ratio)
    first_test_index = round(
        len(dataframes) * (train_ratio + validation_ratio))

    if first_validation_index == 0 or (first_validation_index == first_test_index and validation_ratio > 0):
        # list of dataframes is too short
        dataframe = pd.concat(dataframes)
        return split_dataframe(dataframe, train_ratio, validation_ratio, shuffle)

    if shuffle:
        random.shuffle(dataframes)

    train_dfs = dataframes[:first_validation_index]
    validation_dfs = dataframes[first_validation_index:first_test_index]
    test_dfs = dataframes[first_test_index:]
    train = pd.concat(train_dfs)
    validation = pd.concat(validation_dfs) if len(validation_dfs) else pd.DataFrame(
        columns=train_dfs[0].columns.tolist())
    test = pd.concat(test_dfs) if len(test_dfs) else pd.DataFrame(
        columns=train_dfs[0].columns.tolist())

    # Early check on dataset sizes here should enforce length of each
    return train, validation, test


def split_dataframe(df: pd.DataFrame, train_ratio: float, validation_ratio: float, test_ratio: float = None,
                    shuffle: bool = False):
    """
    Splits dataframe to train, valid and test dfs according to the provided ratios.
    :param df: dataframe to split
    :param train_ratio: float
    :param validation_ratio: float
    :param test_ratio: float
    :param shuffle: bool
    :return: tuple of 3 dataframes
    """
    if shuffle:
        df = df.sample(frac=1)
    first_validation_index = round(df.shape[0] * train_ratio)
    first_test_index = round(df.shape[0] * (train_ratio + validation_ratio))

    train = df.loc[0:first_validation_index]
    validation = df.loc[first_validation_index:first_test_index]
    test = df.loc[first_test_index:]

    return train, validation, test

def write_csv_dataset(df: pd.DataFrame, output_path: str):
    if os.path.isdir(output_path):
        for source_file, results in df.groupby(SOURCE_FILE, as_index=False):
            results = results.drop(SOURCE_FILE, axis=1)
            results.to_csv(os.path.join(output_path, source_file))
    elif os.path.isfile(output_path) and output_path.endswith(".csv"):
        df.to_csv(output_path)


def write_json_dataset(df: pd.DataFrame, output_path: str):
    if os.path.isdir(output_path):
        for source_file, results in df.groupby(SOURCE_FILE, as_index=False):
            results = results.drop(SOURCE_FILE, axis=1)
            results.to_json(os.path.join(
                output_path, source_file), orient='records')
    elif os.path.isfile(output_path) and output_path.endswith(".json"):
        df.to_json(output_path, orient='records')

def map_labels(df: pd.DataFrame, label_col: str, label_map: Dict) -> pd.DataFrame:
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        label_col (str): [description]
        label_map (Dict): [description]

    Returns:
        pd.DataFrame: [description]
    """
    df[label_col] = df[label_col].map(label_map)
    return df


def resample_positives(df: pd.DataFrame, resample_rate: int, label_col: str, pos_label_val: Any, reshuffle=True) -> pd.DataFrame:
    """Given a dataframe, and logic identifying the positive label column, generate multiple copies of rows.
    Generally this is used on the training set with a sparse positive

    Args:
        dfs (pd.DataFrame): Dataframes to process
        resample_rate (int): A positive integer telling us the rate of positives to use. Note that this must be at least 2 to
        actually make copies, because using the value 1 would not actually do anything.
        label_col (str): Will copy rows that match logic for this column name
        pos_label_val (Any): Will copy rows that match logic for the label_col column name to equal this label value.
        reshuffle (bool, optional): Reshuffle the dataframe rows when done. Defaults to True. 
    Returns:
        pd.DataFrame: Dataframes with added rows.
    
    Raises:
        ValueError: resample_rate is not a positive integer
    """
    if not isinstance(resample_rate, int) or resample_rate < 1:
        raise ValueError("Resample rate for positive rows must be a positive integer.")
    elif resample_rate > 1:
        df_to_copy = df[df[label_col] == pos_label_val] 
        df = df.append([df_to_copy] * (resample_rate - 1), ignore_index=True)
    if reshuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    return df

def resample_multilabel_positives(df: pd.DataFrame,
    resample_rate_dict: Dict,
    pos_label_val: Any="1",
    reshuffle=True) -> pd.DataFrame:
    """Given a dataframe, and logic identifying positive label columns and their positive label values, generate multiple copies of rows.
    Generally this is used on the training set with one or more sparse positives.

    For now, all the labels that appear in resample_rate_dict must use the same identified value for a positive label, identified
    as pos_label_val.

    Also note that copies will be made for each label. So if a sample has several positive labels, a separate copy
    will be made for each of the label iteration values. For example:

    {'A': 0, 'B': 1, 'C': 1, 'D': 1} might be a sample, 
    
    and if you had a resample rate dict of: {
        'A': 4,
        'B': 4,
        'C': 10,
        'D': 10
    },

    The you would get 21 new copies of this row (3 + 9 + 9) added to the dataframe.

    Args:
        df (pd.DataFrame): Dataframes to process
        resample_rate_dict (Dict): A dictionary with keys equal to the label columns and values telling us the rate of positives to use for that key.
        Note that this must be at least 2 to actually make copies, because using the value 1 would not actually do anything.
        pos_label_val (Any, optional): Will copies rows that match logic for each label_cols column name to equal this label value. Defaults to "1".
        reshuffle (bool, optional): Reshuffle the dataframe rows when done. Defaults to True. 

    Returns:
        pd.DataFrame: Dataframes with added rows.
    
    Raises:
        ValueError: resample_rate has a value that is not a positive integer
        ValueError: resample_rate has a key that is not a column in the dataframe
    """
    dfs_to_add = []
    for k,v in resample_rate_dict.items():
        if not isinstance(v, int) or v < 1:
            raise ValueError("All resample rates in resample_rate_dict must be positive integers. Value is {} for label {}".format(v,k))
        if k not in list(df.columns):
            raise ValueError("Key {} is in resample_rate_dict but not in dataframe column list".format(k))
        if v > 1:
            df_to_copy = df[df[k] == pos_label_val]
            for copy_index in range(v - 1): #if a resample rate is 2, this adds 1 copy, hence the subtraction
                dfs_to_add.append(df_to_copy)
    
    for df_to_add in dfs_to_add:
        df = df.append(df_to_add, ignore_index=True)
        
    if reshuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    return df
