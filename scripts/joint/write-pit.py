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

THRESHOLDS = {
    "H": 1.1,
    "phi": 1.0,
    "b": 12.0,
}

parser = argparse.ArgumentParser()
parser.add_argument("Y", type=pathlib.Path, help="Training data csv file")
parser.add_argument(
    "train_pred",
    type=pathlib.Path,
    help="npz file of quantile predictions on training points",
)
parser.add_argument(
    "pred",
    type=pathlib.Path,
    help="npz file of quantile predictions on the prediction grid",
)
parser.add_argument("--target", required=True, choices=THRESHOLDS.keys())
parser.add_argument(
    "-o", "--out", type=pathlib.Path, required=True, help="Output npz file."
)
args = parser.parse_args()

Y_train = pd.read_csv(args.Y)
y_actual = Y_train[args.target].to_numpy()

train_pred = np.load(args.train_pred)
pred = np.load(args.pred)

train_pred_quantiles = train_pred["quantiles"]
u_train = quantile_pit(
    train_pred_quantiles.reshape(-1, train_pred_quantiles.shape[-1]),
    train_pred["quantile_levels"],
    y_actual,
)

pred_quantiles = pred["quantiles"]
threshold = THRESHOLDS[args.target]
u_pred = quantile_interpolation(
    pred_quantiles.reshape(-1, pred_quantiles.shape[-1]),
    pred["quantile_levels"],
    threshold=threshold,
)

np.savez(args.out, u_train=u_train, u_pred=u_pred)
