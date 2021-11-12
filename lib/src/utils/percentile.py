import argparse
import os

import pandas as pd

SOURCE_FILE = 'source_base'
SOURCE_DIR = 'source_dir'


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # The config file below specifies values for all arguments listed in this function
    parser.add_argument("--output_dir_suffix", type=str, default='percentiles',
                        help="Suffix to be appended to the input file directory")
    parser.add_argument("--output_col", type=str, default='synthetic_decisions_v1_score_percentile',
                        help="Name of the output column.")
    parser.add_argument("--input_files", type=str, default=None, nargs="+",
                        help="Names of the directories or files.")
    parser.add_argument("--input_col", type=str, default='synthetic_decisions_v1_score',
                        help="Names of the directories or files.")

    return parser


def read_input(input_file: str) -> pd.DataFrame:

    if os.path.isdir(input_file):
        csvs = [t for t in os.listdir(
            input_file) if t.endswith(".csv")]

        dataframes = []

        for csv in csvs:
            fpath = os.path.join(input_file, csv)
            df = pd.read_csv(fpath)
            df[SOURCE_FILE] = csv
            df[SOURCE_DIR] = input_file
            dataframes.append(df)

        df = pd.concat(dataframes)

        return df

    elif os.path.isfile(input_file) and input_file.endswith(".csv"):
        df = pd.read_csv(input_file)
        df[SOURCE_FILE] = os.path.basename(input_file)
        df[SOURCE_DIR] = os.path.dirname(input_file)

        return df

    else:
        raise ValueError("unsupported input format, expected to be either directory with csv files or csv file")


def write_csv_dataset(df: pd.DataFrame, output_path_suffix: str):

    for (dir_name, base_name), results in df.groupby([SOURCE_DIR, SOURCE_FILE], as_index=False):
        results = results.drop([SOURCE_DIR, SOURCE_FILE], axis=1)
        out_dir = os.path.join(dir_name, output_path_suffix)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        results.to_csv(os.path.join(out_dir, base_name))
    

def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    dataframes = []

    for f in args.input_files:
        dataframes.append(read_input(f))

    df = pd.concat(dataframes)

    df[args.output_col] = df[args.input_col].rank(pct=True)

    write_csv_dataset(df, args.output_dir_suffix)


if __name__ == '__main__':
    main()
