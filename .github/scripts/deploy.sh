#!/bin/sh

set -eu

if ! uv pip install --system -r requirements.txt -r notebooks/requirements.txt; then
  exit 1
fi

if ! HEAVYEDGE_GPU_DEVICES=$(python3 scripts/cuda-preflight.py --print-devices); then
  exit 2
fi
export HEAVYEDGE_GPU_DEVICES
if [ -z "${MAKE_JOBS:-}" ] || [ "${MAKE_JOBS}" = "auto" ]; then
  if ! MAKE_JOBS=$(python3 scripts/cuda-preflight.py --print-count); then
    exit 3
  fi
fi
export MAKE_JOBS
if ! python3 scripts/cuda-preflight.py; then
  exit 4
fi

if ! HEAVYEDGE_TEST_MODE=${HEAVYEDGE_TEST_MODE:-} make -j "${MAKE_JOBS}" models notebooks; then
  exit 5
fi

if [ "${PUSH_MODEL:-0}" = "1" ]; then
  if [ -z "${GITHUB_REF_NAME:-}" ]; then
    echo "::error::Missing GITHUB_REF_NAME for model upload." >&2
    exit 6
  fi
  if ! uv pip install --system huggingface_hub; then
    exit 6
  fi
  if ! python upload.py "${GITHUB_REF_NAME}"; then
    exit 6
  fi
fi
