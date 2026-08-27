import argparse
import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="X csv file.")
parser.add_argument(
    "X_index", type=pathlib.Path, help="Mapping from profile names to X indices."
)
parser.add_argument(
    "--test-ratio",
    type=float,
    default=0.2,
    help="Fraction of raw observations reserved as one test set.",
)
parser.add_argument(
    "--draw",
    type=int,
    required=True,
    help="Number of observations drawn with replacement for each remaining X.",
)
parser.add_argument(
    "--num-folds",
    type=int,
    default=5,
    help="Number of non-overlapping train-validation folds.",
)
parser.add_argument("--random-state", type=int, default=42, help="Random state.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output csv file.")
args = parser.parse_args()

if not np.isfinite(args.test_ratio) or not 0 < args.test_ratio < 1:
    parser.error("test ratio must be finite and strictly between 0 and 1")
if args.draw < 1:
    parser.error("draw must be a positive integer")
if args.num_folds < 2:
    parser.error("number of folds must be at least 2")

X = pd.read_csv(args.X, index_col=0)
if "name" not in X.columns:
    parser.error("X must contain a 'name' column")
if X["name"].isna().any():
    parser.error("the 'name' column must not contain missing values")
if not X.index.is_unique:
    parser.error("X row indices must be unique")

X_index = pd.read_csv(args.X_index)
if list(X_index.columns) != ["name", "index"]:
    parser.error("X_index must contain exactly the columns 'name' and 'index'")
if X_index.isna().any().any():
    parser.error("X_index must not contain missing values")
if X_index["name"].duplicated().any():
    parser.error("X_index names must be unique")

index_by_name = X_index.set_index("name")["index"]
missing_names = sorted(set(X["name"]) - set(index_by_name.index))
if missing_names:
    parser.error(
        "X_index does not contain every X profile name; missing: "
        + ", ".join(missing_names)
    )

# Select the test observations directly from all raw rows. Their original row
# indices are retained so they can also be removed from the final refit data.
try:
    remaining_positions, test_positions = train_test_split(
        np.arange(len(X)),
        test_size=args.test_ratio,
        random_state=args.random_state,
    )
except ValueError as error:
    parser.error(f"cannot create the train/test split: {error}")

remaining = X.iloc[remaining_positions].copy()
test = X.iloc[test_positions].copy()
remaining_indices = remaining["name"].map(index_by_name)

all_unique_indices = set(X["name"].map(index_by_name))
remaining_unique_indices = remaining_indices.drop_duplicates().to_numpy()
missing_after_test = sorted(all_unique_indices - set(remaining_unique_indices))
if missing_after_test:
    parser.error(
        "the random test split removed every observation for X indices: "
        + ", ".join(map(str, missing_after_test))
    )

num_unique = len(remaining_unique_indices)
if num_unique < args.num_folds:
    parser.error(
        f"cannot make {args.num_folds} folds from {num_unique} remaining unique X"
    )
if num_unique % args.num_folds:
    parser.error(
        f"{num_unique} remaining unique X values do not divide evenly into "
        f"{args.num_folds} folds"
    )

# Balance the non-test data by drawing exactly the same number of raw
# observations for every unique X. Drawing happens only after test removal.
rng = np.random.default_rng(args.random_state + 1)
drawn_parts = []
for _, group in remaining.groupby(remaining_indices, sort=False):
    positions = rng.choice(len(group), size=args.draw, replace=True)
    drawn_parts.append(group.iloc[positions])
drawn = pd.concat(drawn_parts)
drawn_indices = drawn["name"].map(index_by_name)


def labeled_rows(frame, fold, split):
    split_df = frame.copy()
    split_df.insert(2, "fold", fold)
    split_df.insert(3, "split", split)
    return split_df


fold_dfs = []
cv = KFold(
    n_splits=args.num_folds,
    shuffle=True,
    random_state=args.random_state + 2,
)
for fold, (train_positions, val_positions) in enumerate(
    cv.split(remaining_unique_indices)
):
    train_indices = remaining_unique_indices[train_positions]
    val_indices = remaining_unique_indices[val_positions]
    fold_dfs.extend(
        [
            labeled_rows(drawn[drawn_indices.isin(train_indices)], fold, "train"),
            labeled_rows(drawn[drawn_indices.isin(val_indices)], fold, "val"),
        ]
    )

# Test is shared by all folds and therefore deliberately has no fold value.
fold_dfs.append(labeled_rows(test, pd.NA, "test"))
pd.concat(fold_dfs, axis=0).to_csv(args.out)
