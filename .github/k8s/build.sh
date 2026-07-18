#!/bin/sh

set -eu

if ! uv pip install --system -r requirements.txt -r notebooks/requirements.txt; then
  exit 1
fi

if [ "${PULL_MODEL:-0}" = "1" ]; then
  if [ -z "${MODEL_REVISION:-}" ] || [ -z "${MODEL_REPO_ID:-}" ]; then
    echo "::error::Missing Hugging Face model revision or repository." >&2
    exit 2
  fi
  if [ -z "${HUGGINGFACE_TOKEN:-}" ]; then
    echo "::error::Missing Hugging Face token for model download." >&2
    exit 2
  fi
  if ! uv pip install --system huggingface_hub; then
    exit 2
  fi
  if ! hf download "${MODEL_REPO_ID}" \
      --repo-type model \
      --revision "${MODEL_REVISION}" \
      --token "${HUGGINGFACE_TOKEN}" \
      --local-dir model; then
    exit 2
  fi
fi

if ! HEAVYEDGE_GPU_DEVICES=$(python3 scripts/cuda-preflight.py --print-devices); then
  exit 3
fi
export HEAVYEDGE_GPU_DEVICES
if [ -z "${MAKE_JOBS:-}" ] || [ "${MAKE_JOBS}" = "auto" ]; then
  if ! MAKE_JOBS=$(python3 scripts/cuda-preflight.py --print-count); then
    exit 4
  fi
fi
export MAKE_JOBS
if ! python3 scripts/cuda-preflight.py; then
  exit 5
fi

make_targets="models notebooks"
if [ "${PULL_MODEL:-0}" = "1" ]; then
  make_targets="notebooks"
fi

if ! HEAVYEDGE_TEST_MODE=${DRY_BUILD:-1} make -j "${MAKE_JOBS}" ${make_targets}; then
  exit 6
fi
