#!/bin/sh

set -eu

model_status=0
if [ "${PUSH_MODEL:-0}" = "1" ]; then
  if [ -z "${GITHUB_REF_NAME:-}" ]; then
    echo "::error::Missing GITHUB_REF_NAME for model upload." >&2
    model_status=1
  fi
  if [ "$model_status" -eq 0 ] && ! uv pip install --system huggingface_hub; then
    model_status=1
  fi
  model_metadata_file="${MODEL_UPLOAD_METADATA_FILE:-/tmp/model-upload-metadata.json}"
  if [ "$model_status" -eq 0 ]; then
    rm -f "$model_metadata_file"
    if ! python upload.py "${GITHUB_REF_NAME}" --metadata-file "$model_metadata_file"; then
      model_status=1
    fi
  fi
fi

doc_status=0
if ! sh .github/k8s/push-doc.sh; then
  doc_status=2
fi

exit $((model_status + doc_status))
