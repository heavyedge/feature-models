"""Compute and save marginal CDF values from an MTGPQR model.

Outputs a .npz file with two arrays:
  u_train : (N,) PIT values P(Y <= y_i | x_i) on training data
  u_pred  : (M,) P(Y <= threshold | x) on the prediction grid
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
from pit import quantile_interpolation, quantile_pit

parser = argparse.ArgumentParser()
parser.add_argument("Y", type=pathlib.Path, help="Training data csv file")
parser.add_argument(
    "train_quantiles",
    type=pathlib.Path,
    help="npy file of quantile predictions on training points",
)
parser.add_argument(
    "pred_quantiles",
    type=pathlib.Path,
    help="npy file of quantile predictions on the prediction grid",
)
parser.add_argument("--target", required=True, choices=["H", "phi"])
parser.add_argument(
    "--quantiles",
    type=float,
    nargs="+",
    required=True,
    help="Quantile levels for the model.",
)
parser.add_argument("--threshold", type=float, help="Threshold for PIT computation")
parser.add_argument(
    "-o", "--out", type=pathlib.Path, required=True, help="Output npz file."
)
args = parser.parse_args()

Y_train = pd.read_csv(args.Y)[args.target].to_numpy()
train_quantiles = np.load(args.train_quantiles)
pred_quantiles = np.load(args.pred_quantiles)
quantile_levels = np.array(args.quantiles)


u_train = quantile_pit(
    train_quantiles.reshape(-1, train_quantiles.shape[-1]),
    quantile_levels,
    Y_train,
)

threshold = args.threshold
u_pred = quantile_interpolation(
    pred_quantiles.reshape(-1, pred_quantiles.shape[-1]),
    quantile_levels,
    threshold=threshold,
)

np.savez(args.out, u_train=u_train, u_pred=u_pred)
