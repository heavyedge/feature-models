import argparse
import logging
import pathlib

import numpy as np
import pandas as pd
from progress import ProgressLogger, configure_logging
from scipy.special import ndtr

parser = argparse.ArgumentParser()
parser.add_argument("Y", type=pathlib.Path, help="Training data csv file")
parser.add_argument(
    "pred",
    type=pathlib.Path,
    help="csv file of GPR predictions on training points",
)
parser.add_argument("--index-col", type=int, nargs="+", help="Index column(s) in Y.")
parser.add_argument(
    "--quantiles",
    type=float,
    nargs="+",
    required=True,
    help="Quantile levels (accepted for CLI compatibility with GPQR PIT).",
)
parser.add_argument(
    "-o", "--out", type=pathlib.Path, required=True, help="Output csv file."
)
args = parser.parse_args()
configure_logging()

quantile_levels = np.asarray(args.quantiles, dtype=float)
if (
    not np.isfinite(quantile_levels).all()
    or ((quantile_levels <= 0) | (quantile_levels >= 1)).any()
    or (np.diff(quantile_levels) <= 0).any()
):
    raise ValueError(
        "Quantile levels must be finite, strictly increasing, and in (0, 1)."
    )

logging.info("Reading target data from %s", args.Y)
Y_train = pd.read_csv(args.Y, index_col=args.index_col)
logging.info("Reading GPR predictions from %s", args.pred)
pred = pd.read_csv(args.pred)

required_columns = {
    "index",
    "batch",
    "target",
    "predictive_mean",
    "predictive_std",
}
missing_columns = required_columns.difference(pred.columns)
if missing_columns:
    raise ValueError(
        f"Prediction input is missing required columns: {sorted(missing_columns)}."
    )
if pred["target"].isna().any():
    raise ValueError("Prediction targets must not be missing.")

out_frames = []
for target, target_pred in pred.groupby("target", sort=False):
    if target not in Y_train:
        raise ValueError(f"Target {target!r} is missing from the target data.")

    prediction_indices = pd.to_numeric(target_pred["index"], errors="coerce").to_numpy(
        dtype=float
    )
    if (
        not np.isfinite(prediction_indices).all()
        or not np.equal(prediction_indices, np.floor(prediction_indices)).all()
    ):
        raise ValueError("Prediction indices must be finite integers.")
    prediction_indices = prediction_indices.astype(np.int64)
    if (prediction_indices < 0).any() or (prediction_indices >= len(Y_train)).any():
        raise ValueError("Prediction indices must refer to rows in the target data.")

    means = pd.to_numeric(target_pred["predictive_mean"], errors="coerce").to_numpy(
        dtype=float
    )
    stds = pd.to_numeric(target_pred["predictive_std"], errors="coerce").to_numpy(
        dtype=float
    )
    observations = Y_train[target].to_numpy(dtype=float)[prediction_indices]
    if not np.isfinite(means).all():
        raise ValueError(f"Predictive means for target {target!r} must be finite.")
    if not np.isfinite(stds).all() or (stds <= 0).any():
        raise ValueError(
            f"Predictive standard deviations for target {target!r} "
            "must be finite and positive."
        )
    if not np.isfinite(observations).all():
        raise ValueError(f"Observed values for target {target!r} must be finite.")

    logging.info("Computing PIT for %r (%s rows)", target, f"{len(target_pred):,}")
    progress = ProgressLogger(f"{args.out.stem} | PIT {target}", len(target_pred))
    out = target_pred[["index", "batch"]].copy()
    out.insert(2, "target", target)
    out.insert(3, "sample", 0)
    out["pit"] = ndtr((observations - means) / stds)
    progress.update(len(target_pred))
    out_frames.append(out)

if not out_frames:
    raise ValueError("Prediction input contains no targets.")
logging.info("Writing PIT results to %s", args.out)
pd.concat(out_frames, ignore_index=True).to_csv(args.out, index=False)
logging.info("Done")
