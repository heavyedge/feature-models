import argparse
import pathlib

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="X csv file.")
parser.add_argument(
    "X_index", type=pathlib.Path, help="Mapping from profile names to X indices."
)
parser.add_argument(
    "--index-col",
    type=int,
    nargs="*",
    default=[],
    help="Column indices to exclude from the output.",
)
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output csv file.")
args = parser.parse_args()

X = pd.read_csv(args.X)
if "name" not in X.columns:
    parser.error("X must contain a 'name' column")

X_index = pd.read_csv(args.X_index)
if list(X_index.columns) != ["name", "index"]:
    parser.error("X_index must contain exactly the columns 'name' and 'index'")
if X_index.isna().any().any():
    parser.error("X_index must not contain missing values")
if X_index["name"].duplicated().any():
    parser.error("X_index names must be unique")

index_by_name = X_index.set_index("name")["index"]
X_indices = X["name"].map(index_by_name)
missing_names = sorted(X.loc[X_indices.isna(), "name"].unique())
if missing_names:
    parser.error(
        "X_index does not contain every X profile name; missing: "
        + ", ".join(missing_names)
    )

num_columns = X.shape[1]
if any(index < 0 or index >= num_columns for index in args.index_col):
    parser.error(f"--index-col values must be between 0 and {num_columns - 1}")
if len(set(args.index_col)) != len(args.index_col):
    parser.error("--index-col values must be unique")

index_columns = {X.columns[index] for index in args.index_col}
value_columns = [column for column in X.columns if column not in index_columns]
if not value_columns:
    parser.error("--index-col must leave at least one value column")

X_unique = X.loc[~X_indices.duplicated()].copy()
X_unique[value_columns].to_csv(args.out, index=False)
