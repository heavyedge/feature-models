import argparse
import pathlib
from collections.abc import Iterable

import numpy as np
import pandas as pd

TARGET_COLUMNS = [
    "gap_to_thickness_ratio",
    "capillary_number",
    "cosine_of_contact_angle",
]

parser = argparse.ArgumentParser(description="Construct Xpred grid")
parser.add_argument("X", type=pathlib.Path, help="Observed X csv file.")
parser.add_argument("--target", nargs="*", choices=TARGET_COLUMNS)
parser.add_argument(
    "--start",
    nargs="*",
    type=float,
    default=0.0,
    help="Start value for each minmax-scaled target column.",
)
parser.add_argument(
    "--stop",
    nargs="*",
    type=float,
    default=1.0,
    help="Stop value for each minmax-scaled target column.",
)
parser.add_argument(
    "--ngrid",
    nargs="+",
    type=int,
    help="Number of grid points per target column.",
)
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output csv file.")
args = parser.parse_args()

if args.target is None:
    args.target = []
if len(args.target) != len(set(args.target)):
    raise ValueError("Duplicate targets provided.")
args.target = sorted(args.target, key=lambda x: TARGET_COLUMNS.index(x))
if not isinstance(args.start, Iterable) or len(args.start) == 1:
    args.start = [args.start[0] if isinstance(args.start, list) else args.start] * len(
        args.target
    )
if not isinstance(args.stop, Iterable) or len(args.stop) == 1:
    args.stop = [args.stop[0] if isinstance(args.stop, list) else args.stop] * len(
        args.target
    )
if not isinstance(args.ngrid, Iterable) or len(args.ngrid) == 1:
    args.ngrid = [args.ngrid[0] if isinstance(args.ngrid, list) else args.ngrid] * len(
        args.target
    )

X = pd.read_csv(args.X)
ranges = [
    (
        X[col].min() + s * (X[col].max() - X[col].min()),
        X[col].min() + e * (X[col].max() - X[col].min()),
    )
    for col, s, e in zip(args.target, args.start, args.stop)
]
grids = [np.linspace(r[0], r[1], n) for r, n in zip(ranges, args.ngrid)]
mesh_array = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)

other_columns = [col for col in X.columns if col not in args.target]
other_values = X[other_columns].drop_duplicates().reset_index(drop=True)

grid_shape = mesh_array.shape[:-1]  # e.g. (200, 200)
G = int(np.prod(grid_shape)) if grid_shape else 1
grid_indices = np.indices(grid_shape).reshape(len(grid_shape), -1)

# Non-target TARGET_COLUMNS get indices based on rank in sorted unique values
other_target_columns = [col for col in TARGET_COLUMNS if col not in args.target]
other_target_idx = {
    col: other_values[col]
    .map({v: i for i, v in enumerate(sorted(other_values[col].unique()))})
    .values
    for col in other_target_columns
}

# Build index arrays in TARGET_COLUMNS order
index_arrays = []
index_names = []
for col in TARGET_COLUMNS:
    if col in args.target:
        i = args.target.index(col)
        index_arrays.append(np.tile(grid_indices[i], len(other_values)))
    else:
        index_arrays.append(np.repeat(other_target_idx[col], G))
    index_names.append(col + "_idx")

index = pd.MultiIndex.from_arrays(index_arrays, names=index_names)
Xpred = pd.DataFrame(
    np.tile(mesh_array.reshape(-1, len(args.target)), (len(other_values), 1)),
    columns=args.target,
    index=index,
)

other_values_expanded = pd.DataFrame(
    np.repeat(
        other_values.values, mesh_array.reshape(-1, len(args.target)).shape[0], axis=0
    ),
    columns=other_columns,
    index=index,
)
Xpred = pd.concat([other_values_expanded, Xpred], axis=1)
Xpred = Xpred[X.columns]

Xpred.to_csv(args.out)
