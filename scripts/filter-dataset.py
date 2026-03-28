import argparse
import pathlib

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=pathlib.Path, help="Dataset csv file.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output csv file.")
args = parser.parse_args()

df = pd.read_csv(args.dataset).drop(columns=["Reynolds_number"])
groups = df.groupby("Slurry")

subdfs = []
TARGET_FIELDS = [
    "Feed_slot_height_ratio",
    "Downstream_lip_length_ratio",
    "Upstream_lip_length_ratio",
]
TARGET_VALUES = [0.805, 5.55, 11.11]
for _, subdf in groups:
    subgroups = list(subdf.groupby(TARGET_FIELDS))
    field_values = [subgroup[0] for subgroup in subgroups]
    diff = pd.DataFrame(field_values, columns=TARGET_FIELDS) - TARGET_VALUES
    idx = diff.abs().sum(axis=1).idxmin()
    subdfs.append(subgroups[idx][1])

ret = pd.concat(subdfs, ignore_index=True)
ret.to_csv(args.out, index=False)
