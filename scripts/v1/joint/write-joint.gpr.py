import argparse
import logging
import pathlib

import numpy as np
import pandas as pd
from progress import ProgressLogger, configure_logging

parser = argparse.ArgumentParser()
parser.add_argument(
    "marginal",
    type=pathlib.Path,
    help="csv file of GPR marginal probabilities",
)
parser.add_argument(
    "-o",
    "--out",
    type=pathlib.Path,
    required=True,
    help="Output csv file of joint probabilities.",
)
args = parser.parse_args()
configure_logging()

logging.info("Reading GPR marginal probabilities from %s", args.marginal)
marginal = pd.read_csv(args.marginal)
keys = ["index", "batch", "sample"]
required_columns = set(keys + ["target", "marginal_prob"])
missing_columns = required_columns.difference(marginal.columns)
if missing_columns:
    raise ValueError(
        f"Marginal input is missing required columns: {sorted(missing_columns)}."
    )
if marginal.duplicated(keys + ["target"]).any():
    raise ValueError("Marginal input contains duplicate prediction-target rows.")

marginal["marginal_prob"] = pd.to_numeric(marginal["marginal_prob"], errors="coerce")
probabilities = marginal["marginal_prob"].to_numpy(dtype=float)
if (
    not np.isfinite(probabilities).all()
    or ((probabilities < 0) | (probabilities > 1)).any()
):
    raise ValueError("Marginal probabilities must be finite and in [0, 1].")

marginal_by_target = (
    marginal.set_index(keys + ["target"])["marginal_prob"]
    .unstack("target")
    .sort_index(axis="columns")
)
if marginal_by_target.empty:
    raise ValueError("Marginal input contains no predictions.")
if marginal_by_target.shape[1] < 2:
    raise ValueError("Joint probability requires at least two prediction targets.")
if marginal_by_target.isna().any().any():
    raise ValueError("Each prediction must have a probability for every target.")

logging.info(
    "Computing analytic joint probability for independent GPR targets %s "
    "(%s predictions)",
    marginal_by_target.columns.tolist(),
    f"{len(marginal_by_target):,}",
)
progress = ProgressLogger(
    f"{args.out.stem} | Joint probability", len(marginal_by_target)
)
out = marginal_by_target.index.to_frame(index=False)
out["joint_prob"] = marginal_by_target.prod(axis="columns").to_numpy()
progress.update(len(marginal_by_target))

logging.info("Writing joint probabilities to %s", args.out)
out.to_csv(args.out, index=False)
logging.info("Done")
