#!/bin/sh

set -eu

required_vars="
GITHUB_TOKEN
GITHUB_REPOSITORY
GPU_BUILD_CHECK_RUN_ID
IMAGE_TAG
"

for var_name in ${required_vars}; do
  eval "var_value=\${${var_name}:-}"
  if [ -z "${var_value}" ]; then
    echo "::error::Missing ${var_name} for cd-cleanup dispatch."
    exit 1
  fi
done

payload="$(
  jq -n \
    --arg event_type "cd-cleanup" \
    --arg gpu_build_check_run_id "${GPU_BUILD_CHECK_RUN_ID}" \
    --arg upload_model_check_run_id "${UPLOAD_MODEL_CHECK_RUN_ID:-}" \
    --arg upload_doc_check_run_id "${UPLOAD_DOC_CHECK_RUN_ID:-}" \
    --arg image_tag "${IMAGE_TAG}" \
    '{
      event_type: $event_type,
      client_payload: {
        gpu_build_check_run_id: $gpu_build_check_run_id,
        upload_model_check_run_id: $upload_model_check_run_id,
        upload_doc_check_run_id: $upload_doc_check_run_id,
        image_tag: $image_tag
      }
    }'
)"

curl -fsS \
  -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/${GITHUB_REPOSITORY}/dispatches" \
  -d "${payload}"

echo "Dispatched cd-cleanup for image tag ${IMAGE_TAG}."
