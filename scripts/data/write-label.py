import argparse
import pathlib

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("y", type=pathlib.Path, help="Shape metrics csv file.")
parser.add_argument("--target", nargs="+", help="Target columns.")
parser.add_argument(
    "--threshold", nargs="+", type=float, help="Threshold values for target columns."
)
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output csv file.")
args = parser.parse_args()

y = pd.read_csv(args.y)
label = pd.DataFrame(
    {
        target: (y[target] > threshold)
        for target, threshold in zip(args.target, args.threshold)
    }
)
label.to_csv(args.out, index=False)
