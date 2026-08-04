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
BUILD_CHECK_RUN_ID
CLEANUP_CHECK_RUN_ID
BUILD_CONCLUSION
IMAGE_TAG
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
    --arg build_check_run_id "${BUILD_CHECK_RUN_ID}" \
    --arg cleanup_check_run_id "${CLEANUP_CHECK_RUN_ID}" \
    --arg build_conclusion "${BUILD_CONCLUSION}" \
    --arg deploy_check_run_id "${DEPLOY_CHECK_RUN_ID:-}" \
    --arg deploy_conclusion "${DEPLOY_CONCLUSION:-failure}" \
    --arg doc_deploy_check_run_id "${DOC_DEPLOY_CHECK_RUN_ID:-}" \
    --arg doc_deploy_conclusion "${DOC_DEPLOY_CONCLUSION:-failure}" \
    --arg image_tag "${IMAGE_TAG}" \
    --arg kubernetes_job_name "${KUBERNETES_JOB_NAME:-}" \
    '{
      ref: $ref,
      inputs: {
        build_check_run_id: $build_check_run_id,
        cleanup_check_run_id: $cleanup_check_run_id,
        build_conclusion: $build_conclusion,
        deploy_check_run_id: $deploy_check_run_id,
        deploy_conclusion: $deploy_conclusion,
        doc_deploy_check_run_id: $doc_deploy_check_run_id,
        doc_deploy_conclusion: $doc_deploy_conclusion,
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

echo "Dispatched post-deploy workflows on ref ${GITHUB_DISPATCH_REF}."
