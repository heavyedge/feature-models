import argparse
import pathlib

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Selected input-variable csv file.")
parser.add_argument(
    "X_index", type=pathlib.Path, help="Mapping from profile names to X indices."
)
parser.add_argument("-o", "--out", type=pathlib.Path, required=True)
args = parser.parse_args()

X = pd.read_csv(args.X, index_col=0)
if "name" not in X.columns:
    parser.error("X must contain a 'name' column")
if X["name"].isna().any():
    parser.error("the 'name' column must not contain missing values")

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

# Keep the same first profile name for each X index that write-Xunique.py keeps.
# X_index follows the row order of the merged profile file, so its row position
# is the index consumed by `heavyedge filter`.
unique_names = X.loc[~X_indices.duplicated(), "name"]
profile_position_by_name = pd.Series(
    np.arange(len(X_index), dtype=np.int64), index=X_index["name"]
)
unique_profile_indices = unique_names.map(profile_position_by_name).to_numpy(
    dtype=np.int64
)
args.out.parent.mkdir(parents=True, exist_ok=True)
np.save(args.out, unique_profile_indices)
