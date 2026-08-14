import argparse
import pathlib

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Input variable csv.")
parser.add_argument("shape_features", type=pathlib.Path, help="Shape features csv.")
parser.add_argument("--index-col", type=int, nargs="*", help="Index columns for X.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output csv file.")
args = parser.parse_args()

X = pd.read_csv(args.X, index_col=args.index_col if args.index_col else None)
idx = X.index.get_level_values(0)
y = pd.read_csv(args.shape_features, index_col=0).iloc[idx]
y.set_index(X.index, inplace=True)

y.to_csv(args.out)
