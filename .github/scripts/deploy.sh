#!/bin/sh
set -e

# Build model
if [ "${PUSH_DOC}" = "1" ]; then
    uv pip install --system -r requirements.txt -r notebooks/requirements.txt
else
    uv pip install --system -r requirements.txt
fi

HEAVYEDGE_GPU_DEVICES=$(python3 scripts/cuda-preflight.py --print-devices)
export HEAVYEDGE_GPU_DEVICES
if [ -z "${MAKE_JOBS:-}" ] || [ "${MAKE_JOBS}" = "auto" ]; then
  MAKE_JOBS=$(python3 scripts/cuda-preflight.py --print-count)
fi
export MAKE_JOBS
python3 scripts/cuda-preflight.py

if [ "${PUSH_DOC}" = "1" ]; then
    HEAVYEDGE_TEST_MODE=${HEAVYEDGE_TEST_MODE} make -j "${MAKE_JOBS}" models notebooks
else
    HEAVYEDGE_TEST_MODE=${HEAVYEDGE_TEST_MODE} make -j "${MAKE_JOBS}" models
fi

# Deploy model
if [ "${UPLOAD_TO_HUGGINGFACE}" = "1" ]; then
  if [ -z "${TAG_NAME:-}" ]; then
    echo "TAG_NAME is required to upload to Hugging Face" >&2
    exit 1
  fi
  uv pip install --system huggingface_hub
  if [ -n "${DEPLOY_OUTPUT_TMP_FILE:-}" ]; then
    python upload.py "${TAG_NAME}" --metadata-file "${DEPLOY_OUTPUT_TMP_FILE}"
  else
    python upload.py "${TAG_NAME}"
  fi
fi

# Deploy notebooks
if [ "${PUSH_DOC}" = "1" ]; then
  sh .github/scripts/push-doc.sh
fi
