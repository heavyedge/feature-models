import argparse
import pathlib

import numpy as np
import pandas as pd
from copula import empirical_copula

parser = argparse.ArgumentParser()
parser.add_argument("pit", type=pathlib.Path, help="PIT csv file.")
parser.add_argument("X_pred", type=pathlib.Path, help="Prediction grid csv file.")
parser.add_argument(
    "marginal",
    type=pathlib.Path,
    help="Marginal probability csv files.",
)
parser.add_argument(
    "-o",
    "--out",
    type=pathlib.Path,
    help="Output csv file of joint probability.",
)
args = parser.parse_args()

index_col_pred = [
    "gap_to_thickness_ratio_idx",
    "capillary_number_idx",
    "cosine_of_contact_angle_idx",
]
Xpred = pd.read_csv(args.X_pred, index_col=index_col_pred)

pit = pd.read_csv(args.pit).values
marginal = pd.read_csv(args.marginal).values

# Slurries = Xpred.index.get_level_values("Slurry")

joint_probs = np.empty(len(Xpred), dtype=float)
for cos in Xpred["cosine_of_contact_angle"].unique():
    ok = Xpred["cosine_of_contact_angle"] == cos
    joint_probs[ok] = empirical_copula(pit, marginal[ok])
df = pd.DataFrame(dict(joint_prob=joint_probs), index=Xpred.index)
df.to_csv(args.out, index=True)
