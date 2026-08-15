import argparse
import pathlib

import numpy as np
import pandas as pd
from pit import quantile_pit

parser = argparse.ArgumentParser()
parser.add_argument("Y", type=pathlib.Path, help="Training data csv file")
parser.add_argument(
    "pred",
    type=pathlib.Path,
    help="npy file of quantile predictions on training points",
)
parser.add_argument("--index-col", type=int, nargs="+", help="Index column(s) in Y.")
parser.add_argument("--target", required=True, choices=["H", "phi"])
parser.add_argument(
    "--quantiles",
    type=float,
    nargs="+",
    required=True,
    help="Quantile levels for the model.",
)
parser.add_argument(
    "-o", "--out", type=pathlib.Path, required=True, help="Output csv file."
)
args = parser.parse_args()

Y_train = pd.read_csv(args.Y, index_col=args.index_col)[args.target].to_numpy()
pred = pd.read_csv(
    args.pred, index_col=["index", "batch", "target", "quantile", "sample"]
)

# A prediction file can contain multiple targets.  Select the requested target
# before grouping its quantile predictions into one row per PIT value.
try:
    pred = pred.xs(args.target, level="target")
except KeyError as exc:
    raise ValueError(f"No predictions found for target {args.target!r}.") from exc

pred_values = pred["value"].unstack("quantile").sort_index(axis="columns")
quantile_levels = np.asarray(args.quantiles)
prediction_levels = pred_values.columns.to_numpy()
if not np.array_equal(prediction_levels, quantile_levels):
    raise ValueError(
        "Prediction quantile levels do not match --quantiles: "
        f"got {prediction_levels.tolist()}, expected {quantile_levels.tolist()}."
    )

prediction_indices = pred_values.index.get_level_values("index").to_numpy()
if not np.issubdtype(prediction_indices.dtype, np.integer) or (
    (prediction_indices < 0).any() or (prediction_indices >= len(Y_train)).any()
):
    raise ValueError("Prediction indices must refer to rows in the target data.")

out = pred_values.index.to_frame(index=False)
out["pit"] = quantile_pit(
    pred_values.to_numpy(), quantile_levels, Y_train[prediction_indices]
)
out.to_csv(args.out, index=False)
