import argparse
import pathlib

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Xpred of Rgt and Ca grid")
parser.add_argument("X", type=pathlib.Path, help="Observed X csv file.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output csv file.")
args = parser.parse_args()

COLUMNS = ["Gap_to_thickness_ratio", "Capillary_number"]

X = pd.read_csv(args.X)
slurry_cos = X[["Slurry", "Cos_theta"]].drop_duplicates()

ranges = {col: (X[col].min(), X[col].max()) for col in COLUMNS}
grids = {col: np.linspace(ranges[col][0], ranges[col][1], 200) for col in COLUMNS}
mesh_array = np.stack(np.meshgrid(*grids.values(), indexing="ij"), axis=-1)

grid_shape = mesh_array.shape[:-1]  # e.g. (200, 200)
flat_mesh = mesh_array.reshape(-1, len(COLUMNS))

n_pairs = len(slurry_cos)
n_grid = int(np.prod(grid_shape))
grid_indices = np.indices(grid_shape).reshape(len(grid_shape), -1)

index = pd.MultiIndex.from_arrays(
    [
        np.repeat(slurry_cos["Slurry"].values, n_grid),
    ]
    + [np.tile(grid_indices[i], n_pairs) for i in range(len(grid_shape))],
    names=["Slurry"] + [col + "_idx" for col in COLUMNS],
)

Xpred = pd.DataFrame(
    np.tile(flat_mesh, (n_pairs, 1)),
    index=index,
    columns=COLUMNS,
)
Xpred.insert(
    len(Xpred.columns), "Cos_theta", np.repeat(slurry_cos["Cos_theta"].values, n_grid)
)

Xpred.to_csv(args.out)
