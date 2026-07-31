import argparse
import itertools
import pathlib

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Process variable csv file.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output csv file.")
args = parser.parse_args()

original_df = pd.read_csv(args.X)

df = original_df[
    [
        "Slurry",
        "Gap_to_thickness_ratio",
        "Capillary_number",
        "Feed_slot_height_ratio",
        "Downstream_lip_length_ratio",
        "Upstream_lip_length_ratio",
        "Contact_angle",
    ]
]
groups = df.groupby(
    [
        "Slurry",
        "Feed_slot_height_ratio",
        "Downstream_lip_length_ratio",
        "Upstream_lip_length_ratio",
    ]
)
slurry_idxs = dict()
for i, (slurry, _, _, _) in enumerate(groups.groups.keys()):
    if slurry not in slurry_idxs:
        slurry_idxs[slurry] = [i]
    else:
        slurry_idxs[slurry].append(i)

combinations = list(itertools.product(*slurry_idxs.values()))
die_configs = np.array([k[1:] for k in groups.groups.keys()])
die_config_dists = []
for idxs in combinations:
    die_config = die_configs[list(idxs)]
    die_config_dist = np.linalg.norm(die_config - die_config.mean(axis=0), axis=1)
    die_config_dists.append(die_config_dist.mean())
combination_idx = np.argmin(die_config_dists)
group_idxs = combinations[combination_idx]

indices = []
for i, (_, df) in enumerate(groups):
    if i in group_idxs:
        indices.extend(df.index.tolist())
idxs = np.sort(indices)

original_df.iloc[idxs].to_csv(args.out, index=False)
