import argparse
import os

import pandas as pd


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # The config file below specifies values for all arguments listed in this function
    parser.add_argument("-i", "--input_files", type=str, default=None, nargs="+",
                        help="Names of the directories or individual csv files.")
    parser.add_argument("--by_col", type=str, default='synthetic_decisions_v1_score_percentile',
                        help="Names of the directories or files.")
    parser.add_argument("--extra_col", type=str, nargs="+", default=['text_sample', 'text_sample_id'],
                        help="Column names to include into the output")
    parser.add_argument("-n", "--num", type=int, default=100,
                        help="Number of records to be included in the report, "
                             "a negative value represents positions from the end")
    parser.add_argument("-o", "--output_file", type=str,
                        help="File name (csv) to store results")

    return parser


def read_input(input_file: str) -> pd.DataFrame:

    if os.path.isdir(input_file):
        csvs = [t for t in os.listdir(
            input_file) if t.endswith(".csv")]

        dataframes = []

        for csv in csvs:
            fpath = os.path.join(input_file, csv)
            df = pd.read_csv(fpath)
            df['path'] = fpath
            dataframes.append(df)

        df = pd.concat(dataframes)

        return df

    elif os.path.isfile(input_file) and input_file.endswith(".csv"):
        df = pd.read_csv(input_file)
        df['path'] = os.path.basename(input_file)

        return df

    else:
        raise ValueError("unsupported input format, expected to be either directory with csv files or csv file")


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    dataframes = []

    for f in args.input_files:
        dataframes.append(read_input(f))

    df = pd.concat(dataframes)[list(set(args.extra_col) | set([args.by_col, 'path']))]
    df = df.sort_values(by=[args.by_col], ascending=(args.num >= 0))

    df.head(args.num).to_csv(args.output_file)


if __name__ == '__main__':
    main()
