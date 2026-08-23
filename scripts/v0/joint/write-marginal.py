import argparse
import pathlib

import numpy as np
import pandas as pd
from pit import quantile_interpolation

parser = argparse.ArgumentParser()
parser.add_argument(
    "pred",
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
    "--target",
    help="Only compute the marginal probability for this prediction target.",
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

pred = pd.read_csv(
    args.pred, index_col=["index", "batch", "target", "quantile", "sample"]
)
if args.target is not None:
    available_targets = pred.index.get_level_values("target")
    if args.target not in available_targets:
        raise ValueError(f"Target {args.target!r} is missing from the predictions.")
    pred = pred[available_targets == args.target]

quantile_levels = np.asarray(args.quantiles)
out_frames = []
for target in pred.index.get_level_values("target").unique():
    target_pred = pred.xs(target, level="target")
    pred_values = target_pred["value"].unstack("quantile").sort_index(axis="columns")
    prediction_levels = pred_values.columns.to_numpy()
    if not np.array_equal(prediction_levels, quantile_levels):
        raise ValueError(
            f"Prediction quantile levels for target {target!r} do not match "
            f"--quantiles: got {prediction_levels.tolist()}, "
            f"expected {quantile_levels.tolist()}."
        )

    out = pred_values.index.to_frame(index=False)
    out.insert(2, "target", target)
    out["marginal_prob"] = quantile_interpolation(
        pred_values.to_numpy(),
        quantile_levels,
        threshold=args.threshold,
        device=args.device,
        chunk_size=args.chunk_size,
    )
    out_frames.append(out)

if not out_frames:
    raise ValueError("Prediction input contains no targets.")
pd.concat(out_frames, ignore_index=True).to_csv(args.out, index=False)
