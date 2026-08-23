import argparse
import pathlib

import numpy as np
import pandas as pd
from copula import empirical_copula

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Prediction grid csv file.")
parser.add_argument("pit", type=pathlib.Path, help="PIT csv file.")
parser.add_argument(
    "marginal",
    type=pathlib.Path,
    help="Marginal probability csv files.",
)
parser.add_argument("--index-col", type=int, nargs="+", help="Index columns of X.")
parser.add_argument(
    "-o",
    "--out",
    type=pathlib.Path,
    help="Output csv file of joint probability.",
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

Xpred = pd.read_csv(args.X, index_col=args.index_col)
keys = ["index", "batch", "sample"]


def values_by_target(path, value_column):
    frame = pd.read_csv(path)
    required_columns = set(keys + ["target", value_column])
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing_columns)}."
        )
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

prediction_indices = marginal_values.index.get_level_values("index").to_numpy()
if not np.issubdtype(prediction_indices.dtype, np.integer) or (
    (prediction_indices < 0).any() or (prediction_indices >= len(Xpred)).any()
):
    raise ValueError("Marginal prediction indices must refer to rows in X.")
if "cosine_of_contact_angle" not in Xpred:
    raise ValueError("X must contain a 'cosine_of_contact_angle' column.")

joint_prob = empirical_copula(
    pit_values.to_numpy(),
    marginal_values.to_numpy(),
    chunk_size=args.chunk_size,
    train_chunk_size=args.train_chunk_size,
    device=args.device,
)

out = marginal_values.index.to_frame(index=False)
out["joint_prob"] = joint_prob
out.to_csv(args.out, index=False)
