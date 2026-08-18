import argparse
import pathlib

import numpy as np
import pandas as pd
import torch


def pinball_loss(targets, predictions, quantile_level):
    errors = targets - predictions
    return np.maximum(quantile_level * errors, (quantile_level - 1.0) * errors)


def matched_quantile_levels(pred, quantile_levels, target):
    available_levels = pred["quantile"].unique()
    matched_levels = []
    for level in quantile_levels:
        matches = available_levels[
            np.isclose(available_levels, level, rtol=1e-7, atol=1e-12)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Quantile level {level:g} was not found uniquely in the GPQR "
                f"prediction for target {target!r}."
            )
        matched_levels.append(matches[0])
    return matched_levels


parser = argparse.ArgumentParser(
    description="Calculate pinball loss from multi-output GPR or GPQR predictions."
)
parser.add_argument("pred", type=pathlib.Path, help="Prediction csv file.")
parser.add_argument("y", type=pathlib.Path, help="Target csv file.")
parser.add_argument("--index-col", type=int, nargs="*", help="Index columns for y.")
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

pred = pd.read_csv(args.pred)
y = pd.read_csv(args.y, index_col=args.index_col)

value_columns = (
    {"predictive_mean", "predictive_std"}
    if args.type == "GPR"
    else {"quantile", "value"}
)
required_columns = {"index", "target"} | value_columns
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
    if args.type == "GPR":
        means = pred["predictive_mean"].to_numpy(dtype=float)
        stds = pred["predictive_std"].to_numpy(dtype=float)
    else:
        pred["quantile"] = pd.to_numeric(pred["quantile"], errors="coerce")
        pred["value"] = pd.to_numeric(pred["value"], errors="coerce")
except (TypeError, ValueError) as exc:
    parser.error(f"Invalid prediction values: {exc}")

if not np.isfinite(indices).all() or not np.equal(indices, np.floor(indices)).all():
    parser.error("Prediction indices must be finite integers.")
indices = indices.astype(np.int64)

if args.type == "GPR":
    if not np.isfinite(means).all():
        parser.error("Predictive means must be finite.")
    if not np.isfinite(stds).all() or (stds < 0).any():
        parser.error("Predictive standard deviations must be finite and non-negative.")
    standard_quantiles = (
        torch.distributions.Normal(0.0, 1.0)
        .icdf(torch.as_tensor(quantile_levels, dtype=torch.float64))
        .numpy()
    )
else:
    if not np.isfinite(pred[["quantile", "value"]].to_numpy()).all():
        parser.error("GPQR quantile levels and predictions must be finite.")

records = []
for target, target_pred in pred.groupby("target", sort=False):
    if target not in y.columns:
        parser.error(f"Target column {target!r} is missing from {args.y}.")

    target_row_indices = target_pred.index.to_numpy()
    target_indices = indices[target_row_indices]
    if ((target_indices < 0) | (target_indices >= len(y))).any():
        parser.error(
            f"Prediction indices for target {target!r} must be between 0 and "
            f"{len(y) - 1}, inclusive."
        )

    try:
        target_values = y[target].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        parser.error(f"Target values for {target!r} must be numeric: {exc}")
    selected_targets = target_values[target_indices]
    if not np.isfinite(selected_targets).all():
        parser.error(f"Target values for {target!r} must be finite.")

    if args.type == "GPR":
        prediction_groups = [
            (
                target_indices,
                selected_targets,
                means[target_row_indices]
                + stds[target_row_indices] * standard_quantile,
            )
            for standard_quantile in standard_quantiles
        ]
    else:
        try:
            matched_levels = matched_quantile_levels(
                target_pred, quantile_levels, target
            )
        except ValueError as exc:
            parser.error(str(exc))

        prediction_groups = []
        index_sets = []
        for level in matched_levels:
            level_mask = np.isclose(
                target_pred["quantile"].to_numpy(), level, rtol=1e-7, atol=1e-12
            )
            level_rows = target_pred.index.to_numpy()[level_mask]
            level_indices = indices[level_rows]
            index_sets.append(set(level_indices))
            prediction_groups.append(
                (
                    level_indices,
                    target_values[level_indices],
                    pred.loc[level_rows, "value"].to_numpy(dtype=float),
                )
            )
        if any(index_set != index_sets[0] for index_set in index_sets[1:]):
            parser.error(
                "Every GPQR quantile level must contain the same prediction "
                f"indices for target {target!r}."
            )

    for level, (level_indices, level_targets, predictions) in zip(
        quantile_levels, prediction_groups
    ):
        target_records = pd.DataFrame(
            {
                "index": level_indices,
                "target": target,
                "quantile_level": level,
                "loss": pinball_loss(level_targets, predictions, level),
            }
        )
        # GPQR posterior samples and batched predictions can repeat an
        # observation. Report one loss per output, observation, and quantile.
        records.append(
            target_records.groupby(
                ["index", "target", "quantile_level"], as_index=False
            ).mean()
        )

if records:
    out = pd.concat(records, ignore_index=True)
else:
    out = pd.DataFrame(columns=["index", "target", "quantile_level", "loss"])
out.to_csv(args.out, index=False)
