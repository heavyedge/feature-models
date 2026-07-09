#!/bin/sh

set -eu

uv pip install --system -r requirements.txt -r notebooks/requirements.txt

HEAVYEDGE_GPU_DEVICES=$(python3 scripts/cuda-preflight.py --print-devices)
export HEAVYEDGE_GPU_DEVICES
if [ -z "${MAKE_JOBS:-}" ] || [ "${MAKE_JOBS}" = "auto" ]; then
  MAKE_JOBS=$(python3 scripts/cuda-preflight.py --print-count)
fi
export MAKE_JOBS
python3 scripts/cuda-preflight.py

HEAVYEDGE_TEST_MODE=${HEAVYEDGE_TEST_MODE} make -j "${MAKE_JOBS}" models notebooks
