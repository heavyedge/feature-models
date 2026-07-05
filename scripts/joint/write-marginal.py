import argparse
import pathlib

import numpy as np
from pit import quantile_interpolation

parser = argparse.ArgumentParser()
parser.add_argument(
    "pred_quantiles",
    type=pathlib.Path,
    help="npy file of quantile predictions on the prediction grid",
)
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

pred_quantiles = np.load(args.pred_quantiles)
quantile_levels = np.array(args.quantiles)

threshold = args.threshold
marginal = quantile_interpolation(
    pred_quantiles.reshape(-1, pred_quantiles.shape[-1]),
    quantile_levels,
    threshold=threshold,
)

np.save(args.out, marginal)
