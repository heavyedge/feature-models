import argparse
import pathlib

import numpy as np
import pandas as pd

TARGET_COLUMNS = [
    "Gap_to_thickness_ratio",
    "Capillary_number",
    "Cos_theta",
]

parser = argparse.ArgumentParser(description="Convert Xpred csv to numpy array")
parser.add_argument("Xpred", type=pathlib.Path, help="Xpred csv file.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output npy file.")
args = parser.parse_args()

# Indices for three target columns + 1 for Slurry column
index_col = list(range(len(TARGET_COLUMNS) + 1))
df = pd.read_csv(args.Xpred, index_col=index_col)

np.save(args.out, df.values)
