import argparse
import pathlib

import numpy as np
import pandas as pd
from copula import empirical_copula

parser = argparse.ArgumentParser()
parser.add_argument(
    "X",
    type=pathlib.Path,
    help="Feature csv file (for slurry groupings).",
)
parser.add_argument("X_pred", type=pathlib.Path, help="Prediction grid csv file.")
parser.add_argument(
    "marginal",
    type=pathlib.Path,
    nargs="+",
    help="Marginal distribution npz files.",
)
parser.add_argument(
    "-o",
    "--out",
    type=pathlib.Path,
    help="Output npz file of joint distribution.",
)
args = parser.parse_args()

u_train = np.column_stack([np.load(p)["u_train"] for p in args.marginal])
u_pred_full = np.column_stack([np.load(p)["u_pred"] for p in args.marginal])

Xtrue = pd.read_csv(args.X)
Xpred = pd.read_csv(args.X_pred, index_col=[0, 1, 2])

slurry_levels = Xpred.index.get_level_values("Slurry")

groups = list(Xtrue.groupby("Slurry"))

grids, probs = [], []
for i, (slurry, _) in enumerate(groups):
    xpred = Xpred.xs(slurry, level="Slurry")
    grid_shape = tuple(
        len(xpred.index.get_level_values(j).unique())
        for j in range(xpred.index.nlevels)
    )
    grid = xpred.to_numpy().reshape(*grid_shape, xpred.shape[1])
    grids.append(grid)

    slurry_mask = slurry_levels == slurry
    u_pred_slurry = u_pred_full[slurry_mask]
    joint_prob = empirical_copula(u_train, u_pred_slurry)
    probs.append(joint_prob.reshape(grid_shape))

np.savez(args.out, grids=grids, probs=probs)
