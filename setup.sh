#!/bin/sh

set -e

pip install uv

uv tool install --force 'huggingface_hub[cli]'
export PATH="$(uv tool dir --bin):$PATH"
export HF_TOKEN="${HF_TOKEN:-$HUGGINGFACE_TOKEN}"

mkdir -p ./_data/v1/

uv pip install --system -r requirements.txt -r examples/requirements.txt

hf download heavyedge/profiles --repo-type dataset --revision v1.0.0 --include "v1/process_variables/*.csv" --include "v1/datapackage.json" --local-dir _data/

hf download heavyedge/shape-features --repo-type dataset --revision v1.1.0dev1 --include "v1/shape_features/" --local-dir _data/

# Postprocess data

## Write dimensionless data

uv pip install --system -r libs/profile-dataset/requirements.txt -r libs/profile-dataset/examples/requirements.txt

mkdir -p _data/v1/dimless/mean_profiles
for f in _data/v1/process_variables/mean_profiles/*.csv; do
    out="_data/v1/dimless/mean_profiles/$(basename "$f")"
    papermill libs/profile-dataset/examples/v1/dimless.ipynb - -p pv_path "$f" -p metadata_path _data/v1/datapackage.json -p out_path "$out" > /dev/null 2>&1
done

mkdir -p _data/v1/dimless/all_profiles
for f in _data/v1/process_variables/all_profiles/*.csv; do
    out="_data/v1/dimless/all_profiles/$(basename "$f")"
    papermill libs/profile-dataset/examples/v1/dimless.ipynb - -p pv_path "$f" -p metadata_path _data/v1/datapackage.json -p out_path "$out" > /dev/null 2>&1
done
