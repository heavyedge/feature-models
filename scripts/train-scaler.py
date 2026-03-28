import argparse
import pathlib

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=pathlib.Path, help="Dataset csv file.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output model file.")
args = parser.parse_args()

df = pd.read_csv(args.dataset)
scaler = MinMaxScaler().fit(df)

joblib.dump(scaler, args.out)
