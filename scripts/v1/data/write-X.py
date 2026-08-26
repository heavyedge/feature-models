import argparse
import itertools
import pathlib

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "dimless", type=pathlib.Path, help="Dimensionless variables csv file."
)
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output csv file.")
parser.add_argument(
    "--draw",
    type=int,
    help="Maximum number of raw observations to retain for each X value.",
)
parser.add_argument(
    "--seed", type=int, help="Random seed used when drawing observations."
)
args = parser.parse_args()

if args.draw is not None and args.draw < 1:
    parser.error("--draw must be a positive integer")

df = pd.read_csv(args.dimless)

groups = df.groupby(
    [
        "slurry",
        "feed_slot_height_ratio",
        "downstream_lip_length_ratio",
        "upstream_lip_length_ratio",
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
for i, (_, subdf) in enumerate(groups):
    if i in group_idxs:
        indices.extend(subdf.index.tolist())
idxs = np.sort(indices)

out_df = df.iloc[idxs][
    [
        "name",
        "slurry",
        "gap_to_thickness_ratio",
        "capillary_number",
        "cosine_of_contact_angle",
    ]
]

if args.draw is not None:
    x_columns = [
        "gap_to_thickness_ratio",
        "capillary_number",
        "cosine_of_contact_angle",
    ]
    rng = np.random.default_rng(args.seed)
    sampled_indices = []
    for _, group in out_df.groupby(x_columns, sort=False):
        sampled_indices.extend(
            rng.choice(group.index, size=min(args.draw, len(group)), replace=False)
        )
    out_df = out_df.loc[np.sort(sampled_indices)]

out_df.to_csv(args.out)
