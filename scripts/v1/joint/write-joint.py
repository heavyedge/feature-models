import argparse
import logging
import pathlib

import numpy as np
import pandas as pd
from copula import empirical_copula
from progress import ProgressLogger, configure_logging

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Prediction grid csv file.")
parser.add_argument("pit", type=pathlib.Path, help="PIT csv file.")
parser.add_argument(
    "marginal",
    type=pathlib.Path,
    help="Marginal probability csv file.",
)
parser.add_argument("--index-col", type=int, nargs="+", help="Index columns of X.")
parser.add_argument(
    "-o",
    "--out",
    type=pathlib.Path,
    required=True,
    help="Output csv file of joint probabilities.",
)
parser.add_argument(
    "--device", default="auto", help="Compute device: auto, cpu, or e.g. cuda:0."
)
parser.add_argument(
    "--chunk-size", type=int, default=1024, help="Prediction rows per chunk."
)
parser.add_argument(
    "--train-chunk-size", type=int, default=32768, help="Training rows per chunk."
)
args = parser.parse_args()
configure_logging()

logging.info("Reading prediction grid from %s", args.X)
Xpred = pd.read_csv(args.X, index_col=args.index_col)
keys = ["index", "batch", "sample"]


def values_by_target(path, value_column):
    logging.info("Reading %s values from %s", value_column, path)
    frame = pd.read_csv(path)
    required_columns = set(keys + ["target", value_column])
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing_columns)}."
        )
    if frame.duplicated(keys + ["target"]).any():
        raise ValueError(f"{path} contains duplicate prediction-target rows.")
    return (
        frame.set_index(keys + ["target"])[value_column]
        .unstack("target")
        .sort_index(axis="columns")
    )


pit_values = values_by_target(args.pit, "pit")
marginal_values = values_by_target(args.marginal, "marginal_prob")
missing_pit_targets = marginal_values.columns.difference(pit_values.columns)
if not missing_pit_targets.empty:
    raise ValueError(
        "PIT input is missing marginal targets: " f"{missing_pit_targets.tolist()}."
    )
pit_values = pit_values.reindex(columns=marginal_values.columns)
if pit_values.isna().any().any() or marginal_values.isna().any().any():
    raise ValueError("Each index, batch, and sample must have values for every target.")

pit_array = pit_values.to_numpy(dtype=float)
marginal_array = marginal_values.to_numpy(dtype=float)
if not np.isfinite(pit_array).all() or ((pit_array < 0) | (pit_array > 1)).any():
    raise ValueError("PIT values must be finite and in [0, 1].")
if (
    not np.isfinite(marginal_array).all()
    or ((marginal_array < 0) | (marginal_array > 1)).any()
):
    raise ValueError("Marginal probabilities must be finite and in [0, 1].")

prediction_indices = marginal_values.index.get_level_values("index").to_numpy()
if not np.issubdtype(prediction_indices.dtype, np.integer) or (
    (prediction_indices < 0).any() or (prediction_indices >= len(Xpred)).any()
):
    raise ValueError("Marginal prediction indices must refer to rows in X.")

logging.info(
    "Computing joint probability for %s (%s predictions, %s PIT rows)",
    marginal_values.columns.tolist(),
    f"{len(marginal_values):,}",
    f"{len(pit_values):,}",
)
progress = ProgressLogger(f"{args.out.stem} | Joint probability", len(marginal_values))
joint_prob = empirical_copula(
    pit_array,
    marginal_array,
    chunk_size=args.chunk_size,
    train_chunk_size=args.train_chunk_size,
    device=args.device,
    progress=progress.update,
)

out = marginal_values.index.to_frame(index=False)
out["joint_prob"] = joint_prob
logging.info("Writing joint probabilities to %s", args.out)
out.to_csv(args.out, index=False)
logging.info("Done")
