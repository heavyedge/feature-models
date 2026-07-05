import argparse
import pathlib

import numpy as np
import pandas as pd
from copula import empirical_copula

parser = argparse.ArgumentParser()
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
u_train = np.column_stack([np.load(p)["u_train"] for p in args.marginal])
u_pred_full = np.column_stack([np.load(p)["u_pred"] for p in args.marginal])

Slurries = Xpred.index.get_level_values("Slurry")

joint_probs = []
for slurry in Slurries.unique():
    ok = Slurries == slurry
    u_pred_slurry = u_pred_full[ok]
    joint_probs.append(empirical_copula(u_train, u_pred_slurry))
np.save(args.out, np.concatenate(joint_probs))
