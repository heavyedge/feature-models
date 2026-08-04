#!/bin/sh

set -e

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

## Expand process_variables

mkdir -p _data/v1/process_variables/mean_profiles
python3 -c '
from pathlib import Path
import pandas as pd

pvs = sorted(Path("_data/v1/process_variables/").glob("dataset*.csv"))
features = sorted(Path("_data/v1/shape_features/mean_profiles/").glob("dataset*.csv"))

for pv, feat in zip(pvs, features):
    pv_df = pd.read_csv(pv, dtype=str)
    feature_names = pd.read_csv(feat, dtype=str)["name"]

    expanded_pv = pv_df.set_index("name").reindex(feature_names).reset_index()
    out = Path("_data/v1/process_variables/mean_profiles") / pv.name
    expanded_pv.to_csv(out, index=False)
'

mkdir -p _data/v1/process_variables/profiles
python3 -c '
from pathlib import Path
import pandas as pd

pvs = sorted(Path("_data/v1/process_variables/").glob("dataset*.csv"))
features = sorted(Path("_data/v1/shape_features/profiles/").glob("dataset*.csv"))

for pv, feat in zip(pvs, features):
    pv_df = pd.read_csv(pv, dtype=str)
    feature_names = pd.read_csv(feat, dtype=str)["name"]

    expanded_pv = pv_df.set_index("name").reindex(feature_names).reset_index()
    out = Path("_data/v1/process_variables/profiles") / pv.name
    expanded_pv.to_csv(out, index=False)
'

## Write dimensionless data

uv pip install --system -r libs/profile-dataset/requirements.txt -r libs/profile-dataset/examples/requirements.txt

mkdir -p _data/v1/dimless/mean_profiles
for f in _data/v1/process_variables/mean_profiles/*.csv; do
    out="_data/v1/dimless/mean_profiles/$(basename "$f")"
    papermill libs/profile-dataset/examples/v1/dimless.ipynb - -p pv_path "$f" -p metadata_path _data/v1/datapackage.json -p out_path "$out" > /dev/null 2>&1
done

mkdir -p _data/v1/dimless/profiles
for f in _data/v1/process_variables/profiles/*.csv; do
    out="_data/v1/dimless/profiles/$(basename "$f")"
    papermill libs/profile-dataset/examples/v1/dimless.ipynb - -p pv_path "$f" -p metadata_path _data/v1/datapackage.json -p out_path "$out" > /dev/null 2>&1
done
