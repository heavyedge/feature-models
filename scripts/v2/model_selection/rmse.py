import argparse
import pathlib

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description="Root mean squared error of latent GPR mean predictions."
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

required_columns = {"index", "target", "latent_mean"}
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
    means = pred["latent_mean"].to_numpy(dtype=float)
except (TypeError, ValueError) as exc:
    parser.error(f"Invalid prediction values: {exc}")

if not np.isfinite(indices).all() or not np.equal(indices, np.floor(indices)).all():
    parser.error("Prediction indices must be finite integers.")
indices = indices.astype(np.int64)

if not np.isfinite(means).all():
    parser.error("Latent means must be finite.")

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

    squared_error = (targets - means[target_pred.index]) ** 2
    target_records = pd.DataFrame(
        {"index": target_indices, "target": target, "squared_error": squared_error}
    )
    # Reduce batch predictions before taking the square root, so repeated
    # predictions for an observation produce its root mean squared error.
    target_records = target_records.groupby(["index", "target"], as_index=False).mean()
    target_records["rmse"] = np.sqrt(target_records.pop("squared_error"))
    records.append(target_records)

if records:
    out = pd.concat(records, ignore_index=True)
else:
    out = pd.DataFrame(columns=["index", "target", "rmse"])
out.to_csv(args.out, index=False)
