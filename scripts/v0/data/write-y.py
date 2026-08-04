import argparse
import pathlib

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Input variable csv.")
parser.add_argument("shape_features", type=pathlib.Path, help="Shape features csv.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output csv file.")
args = parser.parse_args()

idx = pd.read_csv(args.X, index_col=0).index
pd.read_csv(args.shape_features).iloc[idx].to_csv(args.out)
