import argparse
import pathlib

import numpy as np
import pandas as pd
import torch


def gpr_quantile_predictions(pred_df, quantile_levels, n_targets):
    required = {"index", "predictive_mean", "predictive_std"}
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
    predictive_means = pred_df["predictive_mean"].to_numpy(dtype=float)
    predictive_stds = pred_df["predictive_std"].to_numpy(dtype=float)
    if not np.isfinite(predictive_means).all():
        raise ValueError("GPR predictive means must be finite.")
    if not np.isfinite(predictive_stds).all() or (predictive_stds < 0).any():
        raise ValueError(
            "GPR predictive standard deviations must be finite and non-negative."
        )

    standard_quantiles = (
        torch.distributions.Normal(0.0, 1.0)
        .icdf(torch.as_tensor(quantile_levels, dtype=torch.float64))
        .numpy()
    )
    predictions = (
        predictive_means[:, None]
        + predictive_stds[:, None] * standard_quantiles
    )
    return indices, predictions


def gpqr_quantile_predictions(pred_df, quantile_levels, n_targets):
    required = {"index", "quantile", "value"}
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
    pred_df["value"] = pd.to_numeric(pred_df["value"], errors="coerce")
    if not np.isfinite(pred_df[["quantile", "value"]].to_numpy()).all():
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

    prediction_groups = [
        pred_df[np.isclose(pred_df["quantile"], level, rtol=1e-7, atol=1e-12)]
        for level in matched_levels
    ]
    index_sets = [set(group["index"]) for group in prediction_groups]
    if any(index_set != index_sets[0] for index_set in index_sets[1:]):
        raise ValueError(
            "Every GPQR quantile level must contain the same prediction indices."
        )

    return [
        (
            group["index"].to_numpy(dtype=np.int64),
            group["value"].to_numpy(dtype=float),
        )
        for group in prediction_groups
    ]


parser = argparse.ArgumentParser(
    description="Calculate pinball loss from saved GPR or GPQR predictions."
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

y_df = pd.read_csv(args.y, index_col=args.index_col)
pred_df = pd.read_csv(args.pred)

if "target" not in pred_df.columns:
    parser.error(f"Prediction column 'target' is missing from {args.pred}.")
if pred_df["target"].isna().any():
    parser.error("Prediction targets must not be missing.")

records = []
for target, target_pred_df in pred_df.groupby("target", sort=False):
    if target not in y_df.columns:
        parser.error(f"Target column {target!r} is missing from {args.y}.")
    try:
        y = y_df[target].to_numpy(dtype=float)
        if args.type == "GPR":
            indices, predictions = gpr_quantile_predictions(
                target_pred_df, quantile_levels, len(y)
            )
            prediction_groups = [
                (indices, predictions[:, quantile_index])
                for quantile_index in range(len(quantile_levels))
            ]
        else:
            prediction_groups = gpqr_quantile_predictions(
                target_pred_df, quantile_levels, len(y)
            )
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    for level, (indices, predictions) in zip(quantile_levels, prediction_groups):
        errors = y[indices] - predictions
        losses = np.maximum(level * errors, (level - 1.0) * errors)
        mean_losses = pd.Series(losses, index=indices).groupby(level=0).mean()
        records.extend(
            {
                "index": index,
                "target": target,
                "quantile_level": level,
                "loss": loss,
            }
            for index, loss in mean_losses.items()
        )

pd.DataFrame.from_records(
    records, columns=["index", "target", "quantile_level", "loss"]
).to_csv(args.out, index=False)
