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
parser.add_argument(
    "--device", default="auto", help="Compute device: auto, cpu, or e.g. cuda:0."
)
parser.add_argument(
    "--chunk-size", type=int, default=262144, help="Rows per compute chunk."
)
args = parser.parse_args()

Y_train = pd.read_csv(args.Y, index_col=args.index_col)
pred = pd.read_csv(
    args.pred, index_col=["index", "batch", "target", "quantile", "sample"]
)

quantile_levels = np.asarray(args.quantiles)
out_frames = []
for target in pred.index.get_level_values("target").unique():
    if target not in Y_train:
        raise ValueError(f"Target {target!r} is missing from the target data.")

    target_pred = pred.xs(target, level="target")
    pred_values = target_pred["value"].unstack("quantile").sort_index(axis="columns")
    prediction_levels = pred_values.columns.to_numpy()
    if not np.array_equal(prediction_levels, quantile_levels):
        raise ValueError(
            f"Prediction quantile levels for target {target!r} do not match "
            f"--quantiles: got {prediction_levels.tolist()}, "
            f"expected {quantile_levels.tolist()}."
        )

    prediction_indices = pred_values.index.get_level_values("index").to_numpy()
    if not np.issubdtype(prediction_indices.dtype, np.integer) or (
        (prediction_indices < 0).any() or (prediction_indices >= len(Y_train)).any()
    ):
        raise ValueError("Prediction indices must refer to rows in the target data.")

    out = pred_values.index.to_frame(index=False)
    out.insert(2, "target", target)
    out["pit"] = quantile_pit(
        pred_values.to_numpy(),
        quantile_levels,
        Y_train[target].to_numpy()[prediction_indices],
        device=args.device,
        chunk_size=args.chunk_size,
    )
    out_frames.append(out)

if not out_frames:
    raise ValueError("Prediction input contains no targets.")
pd.concat(out_frames, ignore_index=True).to_csv(args.out, index=False)
