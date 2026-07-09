#!/bin/sh

set -eu

required_vars="
GITHUB_APP_TOKEN
GITHUB_REPOSITORY
GITHUB_CLEANUP_REF
GPU_BUILD_CHECK_RUN_ID
GPU_BUILD_CONCLUSION
IMAGE_TAG
"

for var_name in ${required_vars}; do
  eval "var_value=\${${var_name}:-}"
  if [ -z "${var_value}" ]; then
    echo "::error::Missing ${var_name} for post-deploy dispatch."
    exit 1
  fi
done

if ! payload="$(
  jq -n \
    --arg ref "${GITHUB_CLEANUP_REF}" \
    --arg gpu_build_check_run_id "${GPU_BUILD_CHECK_RUN_ID}" \
    --arg gpu_build_conclusion "${GPU_BUILD_CONCLUSION}" \
    --arg upload_model_check_run_id "${UPLOAD_MODEL_CHECK_RUN_ID:-}" \
    --arg upload_model_conclusion "${UPLOAD_MODEL_CONCLUSION:-failure}" \
    --arg upload_doc_check_run_id "${UPLOAD_DOC_CHECK_RUN_ID:-}" \
    --arg upload_doc_conclusion "${UPLOAD_DOC_CONCLUSION:-failure}" \
    --arg image_tag "${IMAGE_TAG}" \
    '{
      ref: $ref,
      inputs: {
        gpu_build_check_run_id: $gpu_build_check_run_id,
        gpu_build_conclusion: $gpu_build_conclusion,
        upload_model_check_run_id: $upload_model_check_run_id,
        upload_model_conclusion: $upload_model_conclusion,
        upload_doc_check_run_id: $upload_doc_check_run_id,
        upload_doc_conclusion: $upload_doc_conclusion,
        image_tag: $image_tag
      }
    }'
)"; then
  exit 2
fi

if ! curl -fsS \
  -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${GITHUB_APP_TOKEN}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/${GITHUB_REPOSITORY}/actions/workflows/cd-cleanup.yml/dispatches" \
  -d "${payload}"; then
  exit 3
fi

echo "Dispatched post-deploy workflow for image tag ${IMAGE_TAG} on ref ${GITHUB_CLEANUP_REF}."
