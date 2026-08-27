import argparse
import pathlib
import sys

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
    help="Fraction of unique X values reserved as one fixed outer test set.",
)
parser.add_argument(
    "--num-folds",
    type=int,
    default=5,
    help="Number of inner cross-validation folds within the outer training set.",
)
parser.add_argument("--random-state", type=int, default=42, help="Random state.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output csv file.")
args = parser.parse_args()

if not np.isfinite(args.test_ratio) or not 0 < args.test_ratio < 1:
    parser.error("test ratio must be finite and strictly between 0 and 1")
if args.num_folds < 2:
    parser.error("number of folds must be at least 2")

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

num_unique = len(unique_indices)
candidate_outer_train_sizes = np.arange(
    args.num_folds,
    num_unique,
    args.num_folds,
    dtype=int,
)
if not len(candidate_outer_train_sizes):
    parser.error(
        f"cannot make {args.num_folds} folds from {num_unique} unique X values"
    )

# Equal inner-fold sizes are required because folds are trained as a GP batch.
# Choose the closest outer-test size whose complement is divisible by the number
# of folds. Ties prefer the larger test set.
requested_test_size = args.test_ratio * num_unique
outer_train_size = min(
    candidate_outer_train_sizes,
    key=lambda size: (abs((num_unique - size) - requested_test_size), size),
)
actual_test_size = num_unique - outer_train_size
if not np.isclose(actual_test_size / num_unique, args.test_ratio):
    print(
        f"Adjusted outer test size from {requested_test_size:.2f} to "
        f"{actual_test_size} unique X values so {outer_train_size} outer-training "
        f"values divide evenly into {args.num_folds} folds.",
        file=sys.stderr,
    )

try:
    outer_train_indices, test_indices = train_test_split(
        unique_indices,
        train_size=int(outer_train_size),
        random_state=args.random_state,
    )
except ValueError as error:
    parser.error(f"cannot create the outer train/test split: {error}")


def labeled_rows(indices, fold, split):
    split_df = X[X_indices.isin(indices)].copy()
    split_df.insert(2, "fold", fold)
    split_df.insert(3, "split", split)
    return split_df


fold_dfs = []
inner_cv = KFold(
    n_splits=args.num_folds,
    shuffle=True,
    random_state=args.random_state + 1,
)
for fold, (train_positions, val_positions) in enumerate(
    inner_cv.split(outer_train_indices)
):
    fold_dfs.extend(
        [
            labeled_rows(outer_train_indices[train_positions], fold, "train"),
            labeled_rows(outer_train_indices[val_positions], fold, "val"),
        ]
    )

# Store the outer datasets once. Their synthetic fold value is removed by the
# Make recipes before unbatched outer-model training and evaluation.
fold_dfs.extend(
    [
        labeled_rows(outer_train_indices, -1, "outer_train"),
        labeled_rows(test_indices, -1, "test"),
    ]
)

pd.concat(fold_dfs, axis=0).to_csv(args.out)
