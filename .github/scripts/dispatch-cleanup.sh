#!/bin/sh

set -eu

required_vars="
GITHUB_APP_TOKEN
GITHUB_REPOSITORY
GPU_BUILD_CHECK_RUN_ID
GPU_BUILD_CONCLUSION
CD_CLEANUP_CHECK_RUN_ID
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
    --arg event_type "post-deploy" \
    --arg gpu_build_check_run_id "${GPU_BUILD_CHECK_RUN_ID}" \
    --arg gpu_build_conclusion "${GPU_BUILD_CONCLUSION}" \
    --arg upload_model_check_run_id "${UPLOAD_MODEL_CHECK_RUN_ID:-}" \
    --arg upload_model_conclusion "${UPLOAD_MODEL_CONCLUSION:-failure}" \
    --arg upload_doc_check_run_id "${UPLOAD_DOC_CHECK_RUN_ID:-}" \
    --arg upload_doc_conclusion "${UPLOAD_DOC_CONCLUSION:-failure}" \
    --arg cd_cleanup_check_run_id "${CD_CLEANUP_CHECK_RUN_ID}" \
    --arg image_tag "${IMAGE_TAG}" \
    '{
      event_type: $event_type,
      client_payload: {
        gpu_build_check_run_id: $gpu_build_check_run_id,
        gpu_build_conclusion: $gpu_build_conclusion,
        upload_model_check_run_id: $upload_model_check_run_id,
        upload_model_conclusion: $upload_model_conclusion,
        upload_doc_check_run_id: $upload_doc_check_run_id,
        upload_doc_conclusion: $upload_doc_conclusion,
        cd_cleanup_check_run_id: $cd_cleanup_check_run_id,
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
  "https://api.github.com/repos/${GITHUB_REPOSITORY}/dispatches" \
  -d "${payload}"; then
  exit 3
fi

echo "Dispatched post-deploy for image tag ${IMAGE_TAG}."
