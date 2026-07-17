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

prepare_check_log() {
  source_file="$1"
  output_file="$2"

  # Check output is size-limited. Keep the end of the log, where failures are
  # normally reported, and redact credentials before publishing it.
  python3 - "$source_file" "${CHECK_LOG_MAX_BYTES:-60000}" > "$output_file" <<'PY'
import base64
import os
import sys

source_file, max_bytes_text = sys.argv[1:]
max_bytes = int(max_bytes_text)
with open(source_file, "rb") as file:
    file.seek(0, os.SEEK_END)
    size = file.tell()
    start = max(0, size - max_bytes)
    file.seek(start)
    content = file.read()

if start:
    content = b"[log truncated; showing the final portion]\n" + content

secrets = [
    os.environ.get("GITHUB_APP_TOKEN", "").encode(),
    os.environ.get("HUGGINGFACE_TOKEN", "").encode(),
]
github_token = os.environ.get("GITHUB_APP_TOKEN", "")
if github_token:
    secrets.append(base64.b64encode(f"x-access-token:{github_token}".encode()))

for secret in secrets:
    if secret:
        content = content.replace(secret, b"[REDACTED]")

sys.stdout.write(content.decode("utf-8", errors="replace"))
PY
}

publish_check_log() {
  check_run_id="$1"
  check_title="$2"
  log_file="$3"

  [ -n "$check_run_id" ] || return 0
  if [ ! -f "$log_file" ]; then
    echo "::warning::No log file found for ${check_title}: ${log_file}." >&2
    return 0
  fi

  log_text_file="$(mktemp)"
  payload_file="$(mktemp)"
  response_file="$(mktemp)"

  if ! prepare_check_log "$log_file" "$log_text_file" ||
    ! jq -n \
      --arg title "$check_title" \
      --arg summary "Kubernetes Job phase log. The final conclusion is reported after deployment cleanup." \
      --rawfile text "$log_text_file" \
      '{output: {title: $title, summary: $summary, text: $text}}' > "$payload_file"; then
    echo "::error::Failed to prepare the ${check_title} check log." >&2
    rm -f "$log_text_file" "$payload_file" "$response_file"
    return 1
  fi

  url="https://api.github.com/repos/${GITHUB_REPOSITORY}/check-runs/${check_run_id}"
  if http_status="$(
    curl -sS \
      -o "$response_file" \
      -w "%{http_code}" \
      -X PATCH \
      -H "Accept: application/vnd.github+json" \
      -H "Authorization: Bearer ${GITHUB_APP_TOKEN}" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      -H "Content-Type: application/json" \
      --data-binary "@${payload_file}" \
      "$url"
  )" && [ "$http_status" -ge 200 ] && [ "$http_status" -lt 300 ]; then
    rm -f "$log_text_file" "$payload_file" "$response_file"
    return 0
  fi

  echo "::error::Failed to publish the ${check_title} log; GitHub API returned HTTP ${http_status:-curl-failed} for ${url}." >&2
  if [ -s "$response_file" ]; then
    sed 's/^/GitHub API response: /' "$response_file" >&2
  fi
  rm -f "$log_text_file" "$payload_file" "$response_file"
  return 1
}

publish_kubernetes_job_logs() {
  # The fallback path has no Kubernetes Job logs, so it only dispatches cleanup.
  [ -n "${GPU_BUILD_LOG_FILE:-}" ] || return 0

  publish_check_log \
    "${GPU_BUILD_CHECK_RUN_ID:-}" \
    "GPU build" \
    "$GPU_BUILD_LOG_FILE" || return 1
  publish_check_log \
    "${UPLOAD_MODEL_CHECK_RUN_ID:-}" \
    "Upload model" \
    "${DEPLOY_LOG_FILE:-}" || return 1
  publish_check_log \
    "${UPLOAD_DOC_CHECK_RUN_ID:-}" \
    "Upload document" \
    "${DEPLOY_LOG_FILE:-}" || return 1
}

prepare_cleanup_check_log() {
  cleanup_source_file="$1"
  cleanup_log_file="$2"

  {
    printf '%s\n' '## GitHub App token'
    if [ -f "${GITHUB_APP_TOKEN_LOG_FILE:-}" ]; then
      cat "$GITHUB_APP_TOKEN_LOG_FILE"
    else
      printf '%s\n' 'No GitHub App token log was captured.'
    fi
    printf '\n%s\n' '## Post-deploy dispatch'
    if [ -f "${DEPLOY_DISPATCH_LOG_FILE:-}" ]; then
      cat "$DEPLOY_DISPATCH_LOG_FILE"
    else
      printf '%s\n' 'No post-deploy dispatch log was captured.'
    fi
  } > "$cleanup_source_file"

  CHECK_LOG_MAX_BYTES="${CLEANUP_CHECK_LOG_MAX_BYTES:-50000}" \
    prepare_check_log "$cleanup_source_file" "$cleanup_log_file"
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

if ! publish_kubernetes_job_logs; then
  exit 4
fi

echo "Preparing Kubernetes Job cleanup log."
cleanup_log_source_file="$(mktemp)"
cleanup_log_file="$(mktemp)"
if ! prepare_cleanup_check_log "$cleanup_log_source_file" "$cleanup_log_file"; then
  rm -f "$cleanup_log_source_file" "$cleanup_log_file"
  exit 5
fi

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
    --rawfile kubernetes_job_log "$cleanup_log_file" \
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
        kubernetes_job_log: $kubernetes_job_log
      }
    }'
)"; then
  rm -f "$cleanup_log_source_file" "$cleanup_log_file"
  exit 2
fi
rm -f "$cleanup_log_source_file" "$cleanup_log_file"

echo "Dispatching cd-cleanup workflow."
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
