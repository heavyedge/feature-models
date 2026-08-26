import argparse
import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="X csv file.")
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

unique_names = X["name"].drop_duplicates().to_numpy()

fold_dfs = []
for fold in range(args.num_folds):
    random_state = args.random_state + fold
    try:
        train_names, remaining_names = train_test_split(
            unique_names,
            train_size=train_ratio,
            random_state=random_state,
        )
        val_names, test_names = train_test_split(
            remaining_names,
            train_size=val_ratio / (val_ratio + test_ratio),
            random_state=random_state,
        )
    except ValueError as error:
        parser.error(
            f"cannot split {len(unique_names)} unique X values with the requested "
            f"ratios: {error}"
        )

    split_dfs = []
    for split_index, (split, names) in enumerate(
        (("train", train_names), ("val", val_names), ("test", test_names))
    ):
        split_df = X[X["name"].isin(names)]
        split_df.insert(2, "fold", fold)
        split_df.insert(3, "split", split)
        split_dfs.append(split_df)

    fold_dfs.append(pd.concat(split_dfs, axis=0))

pd.concat(fold_dfs).to_csv(args.out)
