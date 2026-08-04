import argparse
import csv
import pathlib

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "cv",
    type=pathlib.Path,
    nargs="+",
    help="Cross-validation CSV files in long format.",
)
parser.add_argument(
    "index",
    type=int,
    choices=[0, 1],
    help="Metric index: 0=test_mll_loss, 1=test_pinball_loss.",
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
metric_columns = ["test_mll_loss", "test_pinball_loss"]
metric_column = metric_columns[args.index]

cvs = []
for path in args.cv:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: (int(row["epoch"]), int(row["fold"])))
    epochs = sorted({int(row["epoch"]) for row in rows})
    folds = sorted({int(row["fold"]) for row in rows})
    expected_rows = len(epochs) * len(folds)
    if len(rows) != expected_rows:
        raise ValueError(
            f"{path} does not contain a complete epoch/fold grid: "
            f"expected {expected_rows} rows, found {len(rows)}"
        )
    cvs.append(
        np.array([float(row[metric_column]) for row in rows]).reshape(
            len(epochs), len(folds)
        )
    )

mean_losses = [cv.mean(axis=1) for cv in cvs]
best_model_idx = np.argmin([loss.min() for loss in mean_losses])
best_epoch = int(np.median(np.argmin(cvs[best_model_idx], axis=0))) + 1

if args.target == "model":
    with open(args.out, "w") as f:
        f.write(models[best_model_idx])
elif args.target == "epoch":
    with open(args.out, "w") as f:
        f.write(str(best_epoch))
