import argparse
import pathlib

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

parser = argparse.ArgumentParser()
parser.add_argument("X_true", type=pathlib.Path, help="Predictor csv file.")
parser.add_argument("X_pred", type=pathlib.Path, help="Predictor csv file.")
parser.add_argument(
    "--target",
    type=str,
    nargs="+",
    choices=["Gap_to_thickness_ratio", "Capillary_number", "Cos_theta"],
    help="Target name.",
)
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output npy file.")
args = parser.parse_args()

index_col_true = ["Slurry"]
Xtrue = pd.read_csv(args.X_true, index_col=index_col_true)
index_col_pred = [
    "Gap_to_thickness_ratio_idx",
    "Capillary_number_idx",
    "Cos_theta_idx",
    "Slurry",
]
Xpred = pd.read_csv(args.X_pred, index_col=index_col_pred)

Xtrue_Slurries = Xtrue.index.get_level_values("Slurry")
Xpred_Slurries = Xpred.index.get_level_values("Slurry")

simplices = []
for slurry in Xtrue_Slurries.unique():
    xtrue_ok = Xtrue_Slurries == slurry
    xtrue = Xtrue[xtrue_ok][args.target]
    delaunay = Delaunay(xtrue.to_numpy())

    xpred_ok = Xpred_Slurries == slurry
    xpred = Xpred[xpred_ok][args.target]
    simplices.append(delaunay.find_simplex(xpred.to_numpy()) != -1)

np.save(args.out, np.concatenate(simplices))
