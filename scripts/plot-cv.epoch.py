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
    fold_cols = [col for col in cv_df.columns if col.startswith("test_loss_fold")]
    avg_test_loss = cv_df[fold_cols].mean(axis=1)
    label = cv_path.stem.split(".")[1].removesuffix("Mtgpqr")
    plt.plot(cv_df["epoch"], avg_test_loss, label=label)

plt.xlabel("Epoch")
plt.ylabel("Test Loss")
plt.legend()
plt.tight_layout()
plt.savefig(args.out)
