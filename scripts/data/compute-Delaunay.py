import argparse
import pathlib

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

parser = argparse.ArgumentParser()
parser.add_argument("X_true", type=pathlib.Path, help="Predictor csv file.")
parser.add_argument("X_pred", type=pathlib.Path, help="Predictor csv file.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output npy file.")
args = parser.parse_args()

Xtrue = pd.read_csv(args.X_true)
Xpred = pd.read_csv(args.X_pred, index_col=[0, 1, 2]).drop(columns=["Cos_theta"])

simplices = []
for slurry, xtrue in Xtrue.groupby("Slurry"):
    delaunay = Delaunay(xtrue.drop(columns=["Slurry", "Cos_theta"]).to_numpy())
    xpred = Xpred.xs(slurry, level="Slurry")

    grid_shape = tuple(
        len(xpred.index.get_level_values(i).unique())
        for i in range(xpred.index.nlevels)
    )
    xpred = xpred.to_numpy().reshape(*grid_shape, xpred.shape[1])

    simplex = delaunay.find_simplex(xpred.reshape(-1, xpred.shape[-1])) != -1
    simplices.append(simplex.reshape(xpred.shape[:-1]))

np.save(args.out, np.stack(simplices, axis=0))
