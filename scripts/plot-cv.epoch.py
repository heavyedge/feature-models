import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "cv", type=pathlib.Path, nargs="+", help="Cross-validation csv files."
)
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output image file.")
args = parser.parse_args()

for cv_path in args.cv:
    cv_df = pd.read_csv(cv_path)
    label = cv_path.stem.split(".")[1].removesuffix("Mtgpqr")
    plt.plot(cv_df["epoch"], cv_df["test_loss"], label=label)

plt.xlabel("Epoch")
plt.ylabel("Test Loss")
plt.legend()
plt.tight_layout()
plt.savefig(args.out)
