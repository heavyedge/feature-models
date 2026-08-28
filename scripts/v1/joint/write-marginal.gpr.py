import argparse
import logging
import pathlib

import numpy as np
import pandas as pd
from progress import ProgressLogger, configure_logging
from scipy.special import ndtr

parser = argparse.ArgumentParser()
parser.add_argument(
    "pred",
    type=pathlib.Path,
    help="csv file of GPR posterior predictive distributions",
)
parser.add_argument(
    "--threshold",
    type=float,
    required=True,
    help="Threshold for marginal CDF computation",
)
parser.add_argument(
    "--target",
    help="Only compute the marginal probability for this prediction target.",
)
parser.add_argument(
    "-o", "--out", type=pathlib.Path, required=True, help="Output csv file."
)
args = parser.parse_args()
configure_logging()

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
if args.target is not None:
    if args.target not in pred["target"].values:
        raise ValueError(f"Target {args.target!r} is missing from the predictions.")
    pred = pred[pred["target"] == args.target]

out_frames = []
for target, target_pred in pred.groupby("target", sort=False):
    means = pd.to_numeric(target_pred["predictive_mean"], errors="coerce").to_numpy(
        dtype=float
    )
    stds = pd.to_numeric(target_pred["predictive_std"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(means).all():
        raise ValueError(f"Predictive means for target {target!r} must be finite.")
    if not np.isfinite(stds).all() or (stds <= 0).any():
        raise ValueError(
            f"Predictive standard deviations for target {target!r} "
            "must be finite and positive."
        )

    logging.info(
        "Computing marginal probability for %r (%s rows)",
        target,
        f"{len(target_pred):,}",
    )
    progress = ProgressLogger(f"{args.out.stem} | Marginal {target}", len(target_pred))
    out = target_pred[["index", "batch"]].copy()
    out.insert(2, "target", target)
    out.insert(3, "sample", 0)
    out["marginal_prob"] = ndtr((args.threshold - means) / stds)
    progress.update(len(target_pred))
    out_frames.append(out)

if not out_frames:
    raise ValueError("Prediction input contains no targets.")
logging.info("Writing marginal probabilities to %s", args.out)
pd.concat(out_frames, ignore_index=True).to_csv(args.out, index=False)
logging.info("Done")
