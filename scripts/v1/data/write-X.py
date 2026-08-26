import argparse
import itertools
import pathlib

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "dimless", type=pathlib.Path, help="Dimensionless variables csv file."
)
parser.add_argument(
    "X_index", type=pathlib.Path, help="Mapping from profile names to X indices."
)
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output csv file.")
parser.add_argument(
    "--draw",
    type=int,
    help="Number of raw observations to sample with replacement for each X index.",
)
parser.add_argument(
    "--seed", type=int, help="Random seed used when drawing observations."
)
args = parser.parse_args()

if args.draw is not None and args.draw < 1:
    parser.error("--draw must be a positive integer")

df = pd.read_csv(args.dimless)
X_index = pd.read_csv(args.X_index)
if list(X_index.columns) != ["name", "index"]:
    parser.error("X_index must contain exactly the columns 'name' and 'index'")
if X_index["name"].duplicated().any():
    parser.error("X_index names must be unique")

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
    rng = np.random.default_rng(args.seed)
    index_by_name = X_index.set_index("name")["index"]
    missing_names = sorted(set(out_df["name"]) - set(index_by_name.index))
    if missing_names:
        parser.error(
            "X_index does not contain every selected profile name; missing: "
            + ", ".join(missing_names)
        )

    selected = out_df.assign(_X_index=out_df["name"].map(index_by_name))
    sampled_indices = []
    for _, index_group in selected.groupby("_X_index", sort=False):
        names = index_group["name"].drop_duplicates().to_numpy()
        quotas = np.full(len(names), args.draw // len(names), dtype=int)
        remainder = args.draw % len(names)
        if remainder:
            quotas[rng.permutation(len(names))[:remainder]] += 1

        for name, quota in zip(names, quotas):
            if quota == 0:
                continue
            name_indices = index_group.index[index_group["name"] == name].to_numpy()
            sampled_indices.extend(rng.choice(name_indices, size=quota, replace=True))

    out_df = out_df.loc[sampled_indices]

out_df.to_csv(args.out)
