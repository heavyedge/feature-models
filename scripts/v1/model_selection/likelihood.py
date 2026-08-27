import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "joint"))
from pit import quantile_log_density  # noqa: E402


def prediction_indices(pred_df, n_targets):
    indices = pd.to_numeric(pred_df["index"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(indices).all() or not np.equal(indices, np.floor(indices)).all():
        raise ValueError("Prediction indices must be finite integers.")
    indices = indices.astype(np.int64)
    if ((indices < 0) | (indices >= n_targets)).any():
        raise ValueError(
            f"Prediction indices must be between 0 and {n_targets - 1}, inclusive."
        )
    return indices


def gpr_likelihood(pred_df, y):
    required = {"index", "predictive_mean", "predictive_std"}
    missing = required.difference(pred_df.columns)
    if missing:
        raise ValueError(f"GPR prediction is missing columns: {sorted(missing)}")

    indices = prediction_indices(pred_df, len(y))
    means = pred_df["predictive_mean"].to_numpy(dtype=float)
    stds = pred_df["predictive_std"].to_numpy(dtype=float)
    if not np.isfinite(means).all():
        raise ValueError("GPR predictive means must be finite.")
    if not np.isfinite(stds).all() or (stds <= 0).any():
        raise ValueError(
            "GPR predictive standard deviations must be finite and positive."
        )

    standardized = (y[indices] - means) / stds
    log_likelihood = -0.5 * standardized**2 - np.log(stds) - 0.5 * np.log(2.0 * np.pi)
    return indices, log_likelihood


def gpqr_likelihood(pred_df, quantile_levels, y):
    required = {"index", "quantile", "sample", "value"}
    missing = required.difference(pred_df.columns)
    if missing:
        raise ValueError(f"GPQR prediction is missing columns: {sorted(missing)}")

    pred_df = pred_df.copy()
    pred_df["index"] = prediction_indices(pred_df, len(y))
    for column in ("quantile", "sample", "value"):
        pred_df[column] = pd.to_numeric(pred_df[column], errors="coerce")
    if not np.isfinite(pred_df[["quantile", "sample", "value"]].to_numpy()).all():
        raise ValueError("GPQR quantile levels, samples, and values must be finite.")

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

    selected = pred_df[
        np.isclose(
            pred_df["quantile"].to_numpy()[:, None],
            np.asarray(matched_levels)[None, :],
            rtol=1e-7,
            atol=1e-12,
        ).any(axis=1)
    ]
    sample_quantiles = selected.pivot_table(
        index=["index", "sample"], columns="quantile", values="value"
    ).reindex(columns=matched_levels)
    if sample_quantiles.isna().any().any():
        raise ValueError(
            "Every prediction index and posterior sample must contain every "
            "quantile level."
        )

    sample_indices = sample_quantiles.index.get_level_values("index").to_numpy(
        dtype=np.int64
    )
    sample_log_likelihood = quantile_log_density(
        sample_quantiles.to_numpy(dtype=float),
        quantile_levels,
        y[sample_indices],
    )
    sample_results = pd.DataFrame(
        {"index": sample_indices, "log_likelihood": sample_log_likelihood}
    )

    # Evaluate log(mean(exp(log p_s))) stably across posterior samples.
    maxima = sample_results.groupby("index")["log_likelihood"].transform("max")
    sample_results["relative_likelihood"] = np.exp(
        sample_results["log_likelihood"] - maxima
    )
    grouped = sample_results.groupby("index", sort=True)
    log_likelihood = grouped["log_likelihood"].max() + np.log(
        grouped["relative_likelihood"].mean()
    )
    return log_likelihood.index.to_numpy(dtype=np.int64), log_likelihood.to_numpy()


parser = argparse.ArgumentParser(
    description="Evaluate test observations under saved GPR or GPQR predictions."
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
    help="Quantile levels used by GPQR, in strictly increasing order.",
)
parser.add_argument(
    "-o", "--out", type=pathlib.Path, required=True, help="Output csv file."
)
args = parser.parse_args()

if args.type == "GPQR":
    if args.quantile_levels is None:
        parser.error("--quantile-levels is required for GPQR predictions.")
    quantile_levels = np.asarray(args.quantile_levels, dtype=float)
    if (
        not np.isfinite(quantile_levels).all()
        or ((quantile_levels <= 0) | (quantile_levels >= 1)).any()
        or (np.diff(quantile_levels) <= 0).any()
    ):
        parser.error(
            "Quantile levels must be finite, strictly increasing, and in (0, 1)."
        )
else:
    quantile_levels = None

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
        if not np.isfinite(y).all():
            raise ValueError(f"Target values for {target!r} must be finite.")
        if args.type == "GPR":
            indices, log_likelihood = gpr_likelihood(target_pred_df, y)
        else:
            indices, log_likelihood = gpqr_likelihood(
                target_pred_df, quantile_levels, y
            )
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    records.extend(
        {
            "index": index,
            "target": target,
            "likelihood": np.exp(value),
            "negative_log_likelihood": -value,
        }
        for index, value in zip(indices, log_likelihood)
    )

pd.DataFrame.from_records(
    records,
    columns=["index", "target", "likelihood", "negative_log_likelihood"],
).to_csv(args.out, index=False)
