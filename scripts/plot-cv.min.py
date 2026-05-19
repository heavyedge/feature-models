import argparse
import math
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter


def sci_label(value: float, precision: int = 2) -> str:
    if value == 0:
        return f"{0:.{precision}f} × 10^0"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10**exponent)
    return f"{mantissa:.{precision}f} × 10^{exponent}"


parser = argparse.ArgumentParser()
parser.add_argument(
    "cv", type=pathlib.Path, nargs="+", help="Cross-validation csv files."
)
parser.add_argument("--ymin", type=float, help="Minimum y-axis value.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output image file.")
args = parser.parse_args()

names, losses = [], []
for cv_path in args.cv:
    cv_df = pd.read_csv(cv_path)
    fold_cols = [col for col in cv_df.columns if col.startswith("test_loss_fold")]
    avg_test_loss = cv_df[fold_cols].mean(axis=1)
    min_loss = avg_test_loss.min()
    losses.append(min_loss)
    label = cv_path.stem.split(".")[1].removesuffix("Mtgpqr")
    names.append(label)

bars = plt.bar(names, losses)
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        sci_label(height),
        ha="center",
        va="bottom",
    )
plt.xlabel("Model")
plt.ylabel("Minimum Test Loss")
plt.yticks([])
plt.ylim(bottom=float(args.ymin) if args.ymin is not None else 0)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: sci_label(y)))
plt.tight_layout()
plt.savefig(args.out)
