import argparse
import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="X csv file.")
parser.add_argument(
    "X_index", type=pathlib.Path, help="Mapping from profile names to X indices."
)
parser.add_argument(
    "--split-ratio",
    type=float,
    nargs=3,
    default=[0.8, 0.1, 0.1],
    help="Train, validation, test split ratios.",
)
parser.add_argument("--num-folds", type=int, default=10, help="Number of folds.")
parser.add_argument("--random-state", type=int, default=42, help="Random state.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output csv file.")
args = parser.parse_args()

train_ratio, val_ratio, test_ratio = args.split_ratio
if any(ratio <= 0 for ratio in args.split_ratio):
    parser.error("split ratios must be greater than 0")
if not np.isclose(sum(args.split_ratio), 1.0):
    parser.error("split ratios must sum to 1")
if args.num_folds < 1:
    parser.error("number of folds must be at least 1")

X = pd.read_csv(args.X, index_col=0)
if "name" not in X.columns:
    parser.error("X must contain a 'name' column")
if X["name"].isna().any():
    parser.error("the 'name' column must not contain missing values")

X_index = pd.read_csv(args.X_index)
if list(X_index.columns) != ["name", "index"]:
    parser.error("X_index must contain exactly the columns 'name' and 'index'")
if X_index["name"].duplicated().any():
    parser.error("X_index names must be unique")

index_by_name = X_index.set_index("name")["index"]
missing_names = sorted(set(X["name"]) - set(index_by_name.index))
if missing_names:
    parser.error(
        "X_index does not contain every X profile name; missing: "
        + ", ".join(missing_names)
    )

X_indices = X["name"].map(index_by_name)
raw_counts = X_indices.value_counts()
if raw_counts.nunique() != 1:
    parser.error(
        "every X index must have the same number of raw rows; got counts "
        f"{sorted(raw_counts.unique().tolist())}"
    )
unique_indices = X_indices.drop_duplicates().to_numpy()

fold_dfs = []
for fold in range(args.num_folds):
    random_state = args.random_state + fold
    try:
        train_indices, remaining_indices = train_test_split(
            unique_indices,
            train_size=train_ratio,
            random_state=random_state,
        )
        val_indices, test_indices = train_test_split(
            remaining_indices,
            train_size=val_ratio / (val_ratio + test_ratio),
            random_state=random_state,
        )
    except ValueError as error:
        parser.error(
            f"cannot split {len(unique_indices)} unique X values with the requested "
            f"ratios: {error}"
        )

    split_dfs = []
    for split, indices in (
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ):
        split_df = X[X_indices.isin(indices)].copy()
        split_df.insert(2, "fold", fold)
        split_df.insert(3, "split", split)
        split_dfs.append(split_df)

    fold_dfs.append(pd.concat(split_dfs, axis=0))

pd.concat(fold_dfs).to_csv(args.out)
