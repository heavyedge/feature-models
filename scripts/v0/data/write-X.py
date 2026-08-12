import argparse
import itertools
import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument(
    "dimless", type=pathlib.Path, help="Dimensionless variables csv file."
)
parser.add_argument(
    "--split-ratio",
    type=float,
    nargs=3,
    default=[0.8, 0.1, 0.1],
    help="Train, validation, test split ratios.",
)
parser.add_argument("--num-folds", type=int, default=10, help="Number of folds.")
parser.add_argument("--random-state", type=int, default=42, help="Random state.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output csv file.")
args = parser.parse_args()

train_ratio, val_ratio, test_ratio = args.split_ratio
if any(ratio <= 0 for ratio in args.split_ratio):
    parser.error("split ratios must be greater than 0")
if not np.isclose(sum(args.split_ratio), 1.0):
    parser.error("split ratios must sum to 1")
if args.num_folds < 1:
    parser.error("number of folds must be at least 1")

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

fold_dfs = []
for fold in range(1, args.num_folds + 1):
    _out_df = out_df.copy()
    _out_df.insert(2, "fold", fold)
    train_df, remaining_df = train_test_split(
        _out_df,
        train_size=train_ratio,
        random_state=args.random_state + fold - 1,
    )
    val_df, test_df = train_test_split(
        remaining_df,
        train_size=val_ratio / (val_ratio + test_ratio),
        random_state=args.random_state + fold - 1,
    )
    train_df.insert(3, "split", "train")
    val_df.insert(3, "split", "val")
    test_df.insert(3, "split", "test")

    fold_dfs.append(pd.concat([train_df, val_df, test_df], axis=0))

pd.concat(fold_dfs).to_csv(args.out)
