import argparse
import pathlib

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

parser = argparse.ArgumentParser()
parser.add_argument("X_true", type=pathlib.Path, help="Unique X csv file.")
parser.add_argument("X_pred", type=pathlib.Path, help="Grid X csv file.")
parser.add_argument(
    "--grid",
    type=str,
    nargs="+",
    choices=["gap_to_thickness_ratio", "capillary_number", "cosine_of_contact_angle"],
    help="Grid columns to use for Delaunay triangulation.",
)
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output csv file.")
args = parser.parse_args()

Xtrue = pd.read_csv(args.X_true, index_col=[0])
Xpred = pd.read_csv(args.X_pred, index_col=[0, 1, 2])

simplices = np.zeros(len(Xpred), dtype=bool)
for cos in Xtrue["cosine_of_contact_angle"].unique():
    xtrue_ok = Xtrue["cosine_of_contact_angle"] == cos
    xtrue = Xtrue[xtrue_ok][args.grid].drop_duplicates()
    delaunay = Delaunay(xtrue.to_numpy())

    xpred_ok = Xpred["cosine_of_contact_angle"] == cos
    xpred = Xpred[xpred_ok][args.grid]
    simplices[xpred_ok] = delaunay.find_simplex(xpred.to_numpy()) != -1

df = pd.DataFrame(simplices, columns=["in_simplex"], index=Xpred.index)
df.to_csv(args.out, index=True)
