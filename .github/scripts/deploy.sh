#!/bin/sh

set -eu

if [ "${PUSH_MODEL:-0}" = "1" ]; then
  if [ -z "${GITHUB_REF_NAME:-}" ]; then
    echo "::error::Missing GITHUB_REF_NAME for model upload." >&2
    exit 1
  fi
  if ! uv pip install --system huggingface_hub; then
    exit 1
  fi
  if ! python upload.py "${GITHUB_REF_NAME}"; then
    exit 1
  fi
fi

if [ "${PUSH_DOC:-0}" = "1" ]; then
  if [ -z "${GITHUB_REF_NAME:-}" ]; then
    echo "::error::Missing GITHUB_REF_NAME for doc upload." >&2
    exit 2
  fi
  if ! sh .github/scripts/push-doc.sh; then
    exit 2
  fi
fi
