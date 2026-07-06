import argparse
import pathlib

import numpy as np
import pandas as pd
from pit import quantile_pit

parser = argparse.ArgumentParser()
parser.add_argument("Y", type=pathlib.Path, help="Training data csv file")
parser.add_argument(
    "train_quantiles",
    type=pathlib.Path,
    help="npy file of quantile predictions on training points",
)
parser.add_argument("--target", required=True, choices=["H", "phi"])
parser.add_argument(
    "--quantiles",
    type=float,
    nargs="+",
    required=True,
    help="Quantile levels for the model.",
)
parser.add_argument(
    "-o", "--out", type=pathlib.Path, required=True, help="Output npz file."
)
args = parser.parse_args()

Y_train = pd.read_csv(args.Y)[args.target].to_numpy()
train_quantiles = np.load(args.train_quantiles)
quantile_levels = np.array(args.quantiles)


u_train = quantile_pit(
    train_quantiles.reshape(-1, train_quantiles.shape[-1]),
    quantile_levels,
    Y_train,
)

np.save(args.out, u_train)
