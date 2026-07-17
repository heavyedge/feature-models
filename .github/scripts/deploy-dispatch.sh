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
CLEANUP_CHECK_RUN_ID
GPU_BUILD_CONCLUSION
IMAGE_TAG
MODEL_UPLOADED
PUSH_IMAGE
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
    --arg cleanup_check_run_id "${CLEANUP_CHECK_RUN_ID}" \
    --arg gpu_build_conclusion "${GPU_BUILD_CONCLUSION}" \
    --arg upload_model_check_run_id "${UPLOAD_MODEL_CHECK_RUN_ID:-}" \
    --arg upload_model_conclusion "${UPLOAD_MODEL_CONCLUSION:-failure}" \
    --arg upload_doc_check_run_id "${UPLOAD_DOC_CHECK_RUN_ID:-}" \
    --arg upload_doc_conclusion "${UPLOAD_DOC_CONCLUSION:-failure}" \
    --arg image_tag "${IMAGE_TAG}" \
    --arg kubernetes_job_name "${KUBERNETES_JOB_NAME:-}" \
    '{
      ref: $ref,
      inputs: {
        gpu_build_check_run_id: $gpu_build_check_run_id,
        cleanup_check_run_id: $cleanup_check_run_id,
        gpu_build_conclusion: $gpu_build_conclusion,
        upload_model_check_run_id: $upload_model_check_run_id,
        upload_model_conclusion: $upload_model_conclusion,
        upload_doc_check_run_id: $upload_doc_check_run_id,
        upload_doc_conclusion: $upload_doc_conclusion,
        image_tag: $image_tag,
        kubernetes_job_name: $kubernetes_job_name
      }
    }'
)"; then
  exit 2
fi

if ! dispatch_workflow "cd-cleanup.yml" "${cleanup_payload}"; then
  exit 3
fi

if [ "${UPLOAD_DOC_CONCLUSION:-failure}" = "success" ] && [ -n "${DOC_BRANCH:-}" ]; then
  if ! doc_payload="$(
    jq -n \
      --arg ref "${GITHUB_DISPATCH_REF}" \
      --arg doc_branch "${DOC_BRANCH}" \
      --arg push_doc "${PUSH_DOC:-0}" \
      --arg doc_dir "${DOC_DIR:-}" \
      --arg doc_version "${DOC_VERSION:-}" \
      --arg dry_build "${DRY_BUILD:-1}" \
      --arg model_uploaded "${MODEL_UPLOADED}" \
      --arg push_image "${PUSH_IMAGE}" \
      --arg model_revision "${MODEL_REVISION:-}" \
      --arg repo_id "${MODEL_REPO_ID:-}" \
      --arg source_ref "${GITHUB_DISPATCH_REF}" \
      '{
        ref: $ref,
        inputs: {
          doc_branch: $doc_branch,
          push_doc: $push_doc,
          doc_dir: $doc_dir,
          doc_version: $doc_version,
          dry_build: $dry_build,
          model_uploaded: $model_uploaded,
          push_image: $push_image,
          model_revision: $model_revision,
          repo_id: $repo_id,
          source_ref: $source_ref
        }
      }'
  )"; then
    exit 2
  fi

  if ! dispatch_workflow "doc.yml" "${doc_payload}"; then
    exit 3
  fi
else
  echo "Skipping doc.yml dispatch because no documentation branch was pushed."
fi

echo "Dispatched post-deploy workflows for image tag ${IMAGE_TAG} on ref ${GITHUB_DISPATCH_REF}."
