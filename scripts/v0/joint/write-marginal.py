import argparse
import pathlib

import numpy as np
import pandas as pd
from pit import quantile_interpolation

parser = argparse.ArgumentParser()
parser.add_argument(
    "pred_quantiles",
    type=pathlib.Path,
    help="csv file of quantile predictions on the prediction grid",
)
parser.add_argument(
    "--quantiles",
    type=float,
    nargs="+",
    required=True,
    help="Quantile levels for the model.",
)
parser.add_argument(
    "--threshold",
    type=float,
    required=True,
    help="Threshold for marginal CDF computation",
)
parser.add_argument(
    "-o", "--out", type=pathlib.Path, required=True, help="Output csv file."
)
args = parser.parse_args()

pred_quantiles = pd.read_csv(args.pred_quantiles).values
quantile_levels = np.array(args.quantiles)

threshold = args.threshold
marginal = quantile_interpolation(
    pred_quantiles,
    quantile_levels,
    threshold=threshold,
)
df = pd.DataFrame(dict(marginal_prob=marginal))
df.to_csv(args.out, index=False)
