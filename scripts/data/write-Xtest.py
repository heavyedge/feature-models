import argparse
import pathlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser(description="Xtest of Rgt, Ca and cos(theta) grid")
parser.add_argument("X", type=pathlib.Path, help="Observed X csv file.")
parser.add_argument(
    "--start",
    type=int,
    required=True,
    help="Start value of the scaled grid, shared by all axes.",
)
parser.add_argument(
    "--stop",
    type=int,
    required=True,
    help="Stop value of the scaled grid, shared by all axes.",
)
parser.add_argument(
    "--num",
    type=int,
    required=True,
    help="Number of points in the scaled grid, shared by all axes.",
)
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output csv file.")
args = parser.parse_args()

scaler = MinMaxScaler()
X_df = pd.read_csv(args.X).drop(columns=["Slurry"])
X = scaler.fit_transform(X_df.values)

grid = np.meshgrid(
    *[np.linspace(args.start, args.stop, args.num) for _ in range(X.shape[1])],
    indexing="ij",
)
Xtest = scaler.inverse_transform(np.stack(grid, axis=-1).reshape(-1, X.shape[1]))

df = pd.DataFrame(Xtest, columns=X_df.columns)

df.to_csv(args.out, index=False)
