import argparse
import pathlib

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="X csv file.")
parser.add_argument(
    "--index-col",
    type=int,
    nargs="*",
    default=[],
    help="Column indices to exclude from value deduplication and output.",
)
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output csv file.")
args = parser.parse_args()

X = pd.read_csv(args.X)
if "name" not in X.columns:
    parser.error("X must contain a 'name' column")

num_columns = X.shape[1]
if any(index < 0 or index >= num_columns for index in args.index_col):
    parser.error(f"--index-col values must be between 0 and {num_columns - 1}")
if len(set(args.index_col)) != len(args.index_col):
    parser.error("--index-col values must be unique")

X = X.drop_duplicates(subset="name", keep="first")
index_columns = {X.columns[index] for index in args.index_col}
value_columns = [column for column in X.columns if column not in index_columns]
if not value_columns:
    parser.error("--index-col must leave at least one value column")

X_unique = X.drop_duplicates(subset=value_columns, keep="first")
X_unique[value_columns].to_csv(args.out, index=False)
