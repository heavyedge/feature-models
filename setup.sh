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
    "$HOME/.local/bin/hf" download heavyedge/heavyedge-profiles --repo-type dataset --revision v1.0.0rc2 --include "v1/process_variables/*.csv" --include "v1/datapackage.json" --local-dir _data/
) &
pv_pid=$!

(
    "$HOME/.local/bin/hf" download jeesoo9595/heavyedge-features --repo-type dataset --revision v1.0.0a4 --include "v1/shape_features/" --local-dir _data/
) &
features_pid=$!

wait "$requirements_pid"
wait "$pv_pid"
wait "$features_pid"
