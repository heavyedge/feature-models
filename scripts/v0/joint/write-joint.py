import argparse
import pathlib

import numpy as np
import pandas as pd
from copula import empirical_copula

parser = argparse.ArgumentParser()
parser.add_argument("X_pred", type=pathlib.Path, help="Prediction grid csv file.")
parser.add_argument(
    "pit_marginal",
    type=pathlib.Path,
    nargs="+",
    help="PIT and marginal distribution npz files.",
)
parser.add_argument(
    "-o",
    "--out",
    type=pathlib.Path,
    help="Output npy file of joint distribution.",
)
args = parser.parse_args()

index_col_pred = [
    "Gap_to_thickness_ratio_idx",
    "Capillary_number_idx",
    "Cos_theta_idx",
    "Slurry",
]
Xpred = pd.read_csv(args.X_pred, index_col=index_col_pred)
pit = np.column_stack([np.load(p)["pit"] for p in args.pit_marginal])
marginal = np.column_stack([np.load(p)["marginal"] for p in args.pit_marginal])

Slurries = Xpred.index.get_level_values("Slurry")

joint_probs = np.empty(len(Xpred), dtype=float)
for slurry in Slurries.unique():
    ok = Slurries == slurry
    marginal_slurry = marginal[ok]
    joint_probs[ok] = empirical_copula(pit, marginal_slurry)
np.save(args.out, joint_probs)
