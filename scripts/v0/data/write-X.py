import argparse
import pathlib

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("X", type=pathlib.Path, help="Process variable csv file.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output csv file.")
args = parser.parse_args()

df = pd.read_csv(args.X)[
    ["Slurry", "Gap_to_thickness_ratio", "Capillary_number", "Contact_angle"]
]
df["Cos_theta"] = np.cos(np.radians(df["Contact_angle"]))
df.drop("Contact_angle", axis=1).to_csv(args.out, index=False)
