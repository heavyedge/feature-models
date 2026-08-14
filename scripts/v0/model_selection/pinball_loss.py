import argparse
import pathlib

import numpy as np
import pandas as pd
import torch


def gpr_quantile_predictions(pred_df, quantile_levels, n_targets):
    location_column = "loc" if "loc" in pred_df.columns else "mean"
    required = {"index", location_column, "std"}
    missing = required.difference(pred_df.columns)
    if missing:
        raise ValueError(f"GPR prediction is missing columns: {sorted(missing)}")

    indices = pd.to_numeric(pred_df["index"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(indices).all() or not np.equal(indices, np.floor(indices)).all():
        raise ValueError("Prediction indices must be finite integers.")
    indices = indices.astype(np.int64)
    if ((indices < 0) | (indices >= n_targets)).any():
        raise ValueError(
            f"Prediction indices must be between 0 and {n_targets - 1}, inclusive."
        )
    locations = pred_df[location_column].to_numpy(dtype=float)
    stds = pred_df["std"].to_numpy(dtype=float)
    if not np.isfinite(locations).all():
        raise ValueError("GPR locations must be finite.")
    if not np.isfinite(stds).all() or (stds < 0).any():
        raise ValueError("GPR standard deviations must be finite and non-negative.")

    standard_quantiles = (
        torch.distributions.Normal(0.0, 1.0)
        .icdf(torch.as_tensor(quantile_levels, dtype=torch.float64))
        .numpy()
    )
    predictions = locations[:, None] + stds[:, None] * standard_quantiles
    return indices, predictions


def gpqr_quantile_predictions(pred_df, quantile_levels, target, n_targets):
    required = {"index", "quantile", target}
    missing = required.difference(pred_df.columns)
    if missing:
        raise ValueError(f"GPQR prediction is missing columns: {sorted(missing)}")

    pred_df = pred_df.copy()
    pred_df["index"] = pd.to_numeric(pred_df["index"], errors="coerce")
    if (
        not np.isfinite(pred_df["index"]).all()
        or not np.equal(pred_df["index"], np.floor(pred_df["index"])).all()
    ):
        raise ValueError("Prediction indices must be finite integers.")
    pred_df["index"] = pred_df["index"].astype(np.int64)
    if ((pred_df["index"] < 0) | (pred_df["index"] >= n_targets)).any():
        raise ValueError(
            f"Prediction indices must be between 0 and {n_targets - 1}, inclusive."
        )
    pred_df["quantile"] = pd.to_numeric(pred_df["quantile"], errors="coerce")
    pred_df[target] = pd.to_numeric(pred_df[target], errors="coerce")
    if not np.isfinite(pred_df[["quantile", target]].to_numpy()).all():
        raise ValueError("GPQR quantile levels and predictions must be finite.")

    available_levels = pred_df["quantile"].unique()
    matched_levels = []
    for level in quantile_levels:
        matches = available_levels[
            np.isclose(available_levels, level, rtol=1e-7, atol=1e-12)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Quantile level {level:g} was not found in the GPQR prediction."
            )
        matched_levels.append(matches[0])

    mean_predictions = pred_df.groupby(["index", "quantile"], sort=False)[target].mean()
    index_sets = [
        set(mean_predictions.xs(level, level="quantile").index)
        for level in matched_levels
    ]
    if any(index_set != index_sets[0] for index_set in index_sets[1:]):
        raise ValueError(
            "Every GPQR quantile level must contain the same prediction indices."
        )

    indices = np.asarray(sorted(index_sets[0]), dtype=np.int64)
    predictions = np.column_stack(
        [
            mean_predictions.xs(level, level="quantile").reindex(indices).to_numpy()
            for level in matched_levels
        ]
    )
    return indices, predictions


parser = argparse.ArgumentParser(
    description="Calculate pinball loss from saved GPR or GPQR predictions."
)
parser.add_argument("pred", type=pathlib.Path, help="Prediction csv file.")
parser.add_argument("y", type=pathlib.Path, help="Target csv file.")
parser.add_argument("--index-col", type=int, nargs="*", help="Index columns for y.")
parser.add_argument("--target", type=str, required=True, help="Target variable name.")
parser.add_argument(
    "--type", type=str, choices=["GPR", "GPQR"], required=True, help="Prediction type."
)
parser.add_argument(
    "--quantile-levels",
    type=float,
    nargs="+",
    required=True,
    help="Quantile levels to evaluate.",
)
parser.add_argument(
    "-o", "--out", type=pathlib.Path, required=True, help="Output csv file."
)
args = parser.parse_args()

quantile_levels = np.asarray(args.quantile_levels, dtype=float)
if (
    not np.isfinite(quantile_levels).all()
    or ((quantile_levels <= 0) | (quantile_levels >= 1)).any()
):
    parser.error("Quantile levels must be finite and strictly between 0 and 1.")
if len(np.unique(quantile_levels)) != len(quantile_levels):
    parser.error("Quantile levels must not contain duplicates.")

y_df = pd.read_csv(args.y, index_col=args.index_col)
if args.target not in y_df.columns:
    parser.error(f"Target column {args.target!r} is missing from {args.y}.")
y = y_df[args.target].to_numpy(dtype=float)
pred_df = pd.read_csv(args.pred)

try:
    if args.type == "GPR":
        indices, predictions = gpr_quantile_predictions(
            pred_df, quantile_levels, len(y)
        )
    else:
        indices, predictions = gpqr_quantile_predictions(
            pred_df, quantile_levels, args.target, len(y)
        )
except (KeyError, TypeError, ValueError) as exc:
    parser.error(str(exc))

errors = y[indices, None] - predictions
losses = np.maximum(
    quantile_levels[None, :] * errors,
    (quantile_levels[None, :] - 1.0) * errors,
).mean(axis=0)
pd.DataFrame({"quantile_level": quantile_levels, "loss": losses}).to_csv(
    args.out, index=False
)
