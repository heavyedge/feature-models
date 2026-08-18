import argparse
import pathlib

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description="Negative log predictive density loss of GPR predictions."
)
parser.add_argument("pred", type=pathlib.Path, help="Prediction csv file.")
parser.add_argument("y", type=pathlib.Path, help="Target csv file.")
parser.add_argument("--index-col", type=int, nargs="*", help="Index columns for y.")
parser.add_argument(
    "-o", "--out", type=pathlib.Path, required=True, help="Output csv file."
)
args = parser.parse_args()

pred = pd.read_csv(args.pred)
y = pd.read_csv(args.y, index_col=args.index_col)

required_columns = {"index", "target", "predictive_mean", "predictive_std"}
missing_columns = required_columns.difference(pred.columns)
if missing_columns:
    parser.error(
        "Prediction csv is missing required column(s): "
        + ", ".join(sorted(missing_columns))
    )

if pred["target"].isna().any():
    parser.error("Prediction targets must not be missing.")

try:
    indices = pd.to_numeric(pred["index"], errors="coerce").to_numpy(dtype=float)
    means = pred["predictive_mean"].to_numpy(dtype=float)
    stds = pred["predictive_std"].to_numpy(dtype=float)
except (TypeError, ValueError) as exc:
    parser.error(f"Invalid prediction values: {exc}")

if not np.isfinite(indices).all() or not np.equal(indices, np.floor(indices)).all():
    parser.error("Prediction indices must be finite integers.")
indices = indices.astype(np.int64)

if not np.isfinite(means).all():
    parser.error("Predictive means must be finite.")
if not np.isfinite(stds).all() or (stds <= 0).any():
    parser.error("Predictive standard deviations must be finite and positive.")

records = []
for target, target_pred in pred.groupby("target", sort=False):
    if target not in y.columns:
        parser.error(f"Target column {target!r} is missing from {args.y}.")

    target_indices = indices[target_pred.index]
    if ((target_indices < 0) | (target_indices >= len(y))).any():
        parser.error(
            f"Prediction indices for target {target!r} must be between 0 and "
            f"{len(y) - 1}, inclusive."
        )

    try:
        targets = y[target].to_numpy(dtype=float)[target_indices]
    except (TypeError, ValueError) as exc:
        parser.error(f"Target values for {target!r} must be numeric: {exc}")
    if not np.isfinite(targets).all():
        parser.error(f"Target values for {target!r} must be finite.")

    target_means = means[target_pred.index]
    target_stds = stds[target_pred.index]
    nlpd = (
        0.5 * np.log(2.0 * np.pi * target_stds**2)
        + 0.5 * ((targets - target_means) / target_stds) ** 2
    )

    # A batched prediction may contain several posterior draws for the same
    # observation.  Keep one score per observation and target by averaging
    # those draws, matching the per-index reporting used by the benchmarks.
    target_records = pd.DataFrame(
        {"index": target_indices, "target": target, "nlpd": nlpd}
    )
    records.append(target_records.groupby(["index", "target"], as_index=False).mean())

if records:
    out = pd.concat(records, ignore_index=True)
else:
    out = pd.DataFrame(columns=["index", "target", "nlpd"])
out.to_csv(args.out, index=False)
