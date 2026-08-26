import argparse
import pathlib

import pandas as pd

X_COLUMNS = [
    "slurry",
    "gap_to_thickness_ratio",
    "capillary_number",
    "cosine_of_contact_angle",
    "feed_slot_height_ratio",
    "downstream_lip_length_ratio",
    "upstream_lip_length_ratio",
]


parser = argparse.ArgumentParser()
parser.add_argument(
    "mean_profiles",
    type=pathlib.Path,
    nargs="+",
    help="Mean-profile csv files.",
)
parser.add_argument("-o", "--out", type=pathlib.Path, required=True)
args = parser.parse_args()

frames = [
    pd.read_csv(path, dtype=str).assign(
        name=lambda df, stem=path.stem: stem + "/" + df["name"]
    )
    for path in sorted(args.mean_profiles)
]
df = pd.concat(frames, ignore_index=True)

missing_columns = [column for column in X_COLUMNS if column not in df.columns]
if missing_columns:
    parser.error(
        "mean-profile csv files are missing required columns: "
        + ", ".join(missing_columns)
    )
if df["name"].duplicated().any():
    parser.error("mean-profile names must be unique after adding dataset prefixes")

index, _ = pd.factorize(pd.MultiIndex.from_frame(df[X_COLUMNS]), sort=False)
pd.DataFrame({"name": df["name"], "index": index}).to_csv(args.out, index=False)
