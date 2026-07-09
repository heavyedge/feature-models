#!/bin/sh

set -eu

dispatch_workflow() {
  workflow_file="$1"
  payload="$2"
  url="https://api.github.com/repos/${GITHUB_REPOSITORY}/actions/workflows/${workflow_file}/dispatches"
  response_file="$(mktemp)"

  if http_status="$(
    curl -sS \
      -o "${response_file}" \
      -w "%{http_code}" \
      -X POST \
      -H "Accept: application/vnd.github+json" \
      -H "Authorization: Bearer ${GITHUB_APP_TOKEN}" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "${url}" \
      -d "${payload}"
  )" && [ "${http_status}" -ge 200 ] && [ "${http_status}" -lt 300 ]; then
    rm -f "${response_file}"
    return 0
  fi

  echo "::error::Failed to dispatch ${workflow_file} on ref ${GITHUB_DISPATCH_REF}; GitHub API returned HTTP ${http_status:-curl-failed} for ${url}." >&2
  if [ -s "${response_file}" ]; then
    sed 's/^/GitHub API response: /' "${response_file}" >&2
  fi
  rm -f "${response_file}"
  return 1
}

required_vars="
GITHUB_APP_TOKEN
GITHUB_REPOSITORY
GITHUB_DISPATCH_REF
GPU_BUILD_CHECK_RUN_ID
GPU_BUILD_CONCLUSION
IMAGE_TAG
MODEL_UPLOADED
"

for var_name in ${required_vars}; do
  eval "var_value=\${${var_name}:-}"
  if [ -z "${var_value}" ]; then
    echo "::error::Missing ${var_name} for post-deploy dispatch."
    exit 1
  fi
done

if ! cleanup_payload="$(
  jq -n \
    --arg ref "${GITHUB_DISPATCH_REF}" \
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

if ! image_payload="$(
  jq -n \
    --arg ref "${GITHUB_DISPATCH_REF}" \
    --arg MODEL_UPLOADED "${MODEL_UPLOADED:-}" \
    --arg MODEL_REVISION "${MODEL_REVISION:-}" \
    --arg MODEL_REPO_ID "${MODEL_REPO_ID:-}" \
    '{
      ref: $ref,
      inputs: {
        model_uploaded: $MODEL_UPLOADED,
        model_revision: $MODEL_REVISION,
        repo_id: $MODEL_REPO_ID
      }
    }'
)"; then
  exit 2
fi

if ! dispatch_workflow "cd-cleanup.yml" "${cleanup_payload}"; then
  exit 3
fi

if ! dispatch_workflow "image.yml" "${image_payload}"; then
  exit 3
fi

echo "Dispatched post-deploy workflows for image tag ${IMAGE_TAG} on ref ${GITHUB_DISPATCH_REF}."
