import argparse
import pathlib

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description="Quantify quantile crossing in GPQR predictions."
)
parser.add_argument("pred", type=pathlib.Path, help="Prediction csv file.")
parser.add_argument(
    "-o", "--out", type=pathlib.Path, required=True, help="Output csv file."
)
args = parser.parse_args()

pred = pd.read_csv(
    args.pred, index_col=["index", "batch", "target", "quantile", "sample"]
)
records = []

# Keep the prediction index and quantile-difference axes, while reducing the
# batch and sample axes for each prediction index independently.
for (index, target), index_pred in pred.groupby(level=["index", "target"], sort=False):
    quantile_diffs = []
    quantiles = None
    for _, group in index_pred.groupby(
        level=["batch", "sample"], sort=False, dropna=False
    ):
        group = group.sort_index(level="quantile")
        group_quantiles = group.index.get_level_values("quantile").to_numpy()
        if quantiles is None:
            quantiles = group_quantiles
        elif not np.array_equal(quantiles, group_quantiles):
            raise ValueError(
                f"Inconsistent quantile levels for prediction index {index!r}."
            )
        quantile_diffs.append(np.diff(group["value"].to_numpy()))

    # (batch * sample, Q-1)
    quantile_diff = np.stack(quantile_diffs, axis=0)
    crossing = quantile_diff < 0
    crossing_size = np.where(crossing, -quantile_diff, 0)

    # (Q-1,): aggregate only over batch and sample.
    crossing_rate = crossing.mean(axis=0)
    mean_crossing = crossing_size.mean(axis=0)
    max_crossing = crossing_size.max(axis=0)

    for q1, q2, target_rate, target_mean, target_maximum in zip(
        quantiles[:-1], quantiles[1:], crossing_rate, mean_crossing, max_crossing
    ):
        records.append(
            {
                "index": index,
                "q1": q1,
                "q2": q2,
                "target": target,
                "crossing_rate": target_rate,
                "mean_crossing": target_mean,
                "max_crossing": target_maximum,
            }
        )

out = pd.DataFrame.from_records(records).set_index(["index", "q1", "q2", "target"])
out.to_csv(args.out)
