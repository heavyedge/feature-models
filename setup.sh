#!/bin/sh

pip install uv
curl -LsSf https://hf.co/cli/install.sh | bash
"$HOME/.local/bin/hf" auth login --token "$HUGGINGFACE_TOKEN"

mkdir -p ./_data/v1/

(
    uv pip install --system -r requirements.txt -r examples/requirements.txt
) &
requirements_pid=$!

(
    "$HOME/.local/bin/hf" download heavyedge/profiles --repo-type dataset --revision v1.0.0rc3 --include "v1/process_variables/*.csv" --include "v1/datapackage.json" --local-dir _data/
) &
pv_pid=$!

(
    "$HOME/.local/bin/hf" download heavyedge/shape-features --repo-type dataset --revision v1.0.0b1 --include "v1/shape_features/" --local-dir _data/
) &
features_pid=$!

wait "$requirements_pid"
wait "$pv_pid"
wait "$features_pid"

# Postprocess data

python3 -c '
from pathlib import Path
import pandas as pd

pvs = sorted(Path("_data/v1/process_variables").glob("*.csv"))
df = pd.concat(
    [pd.read_csv(f, dtype=str).assign(name=lambda df: f"{f.stem}/" + df["name"]) for f in pvs],
    ignore_index=True,
)
df.to_csv("_data/v1/process_variables.csv", index=False)
'

uv pip install --system -r libs/profile-dataset/requirements.txt -r libs/profile-dataset/examples/requirements.txt
papermill libs/profile-dataset/examples/v1/dimless.ipynb - -p pv_path _data/v1/process_variables.csv -p metadata_path _data/v1/datapackage.json -p out_path "_data/v1/dimless.csv" > /dev/null 2>&1
