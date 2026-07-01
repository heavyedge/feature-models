import argparse
import pathlib

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "cv",
    type=pathlib.Path,
    nargs="+",
    help="Cross-validation CSV files.",
)
parser.add_argument(
    "--target",
    required=True,
    choices=["model", "epoch"],
    help="Target config.",
)
parser.add_argument(
    "-o",
    "--out",
    type=pathlib.Path,
    help="Output file for the best configuration.",
)
args = parser.parse_args()

models = [f.stem.split(".")[1] for f in args.cv]
cvs = [pd.read_csv(f).mean(axis=1) for f in args.cv]

best_model_idx = np.argmax([cv.max() for cv in cvs])
best_epoch = cvs[best_model_idx].argmax() + 1

if args.target == "model":
    with open(args.out, "w") as f:
        f.write(models[best_model_idx])
elif args.target == "epoch":
    with open(args.out, "w") as f:
        f.write(str(best_epoch))
