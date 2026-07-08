#!/bin/sh

set -eu

action="${1:-}"
check_name="${GITHUB_CHECK_RUN_NAME:-GPU run}"
check_id_file="${GITHUB_CHECK_RUN_ID_FILE:-/tmp/heavyedge-github-check-run-id}"

if [ "${action}" != "started" ] && [ "${action}" != "succeeded" ] && [ "${action}" != "failed" ]; then
  echo "Usage: github-check-run.sh <started|succeeded|failed>" >&2
  exit 2
fi

if [ -z "${GH_APP_ID:-}" ] || [ -z "${GH_APP_PRIVATE_KEY:-}" ] || [ -z "${GITHUB_REPOSITORY:-}" ] || [ -z "${GIT_SHA:-}" ]; then
  echo "GitHub App check-run environment is incomplete; skipping ${check_name} ${action} update." >&2
  exit 0
fi

. .github/scripts/github-app.sh

owner="${GITHUB_REPOSITORY%/*}"
repo="${GITHUB_REPOSITORY#*/}"
token="$(github_app_create_installation_token '{"checks":"write"}')"
api_url="https://api.github.com/repos/${owner}/${repo}/check-runs"

github_get_commit() {
  curl -sS \
    -o /dev/null \
    -w "%{http_code}" \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${token}" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/${owner}/${repo}/commits/${GIT_SHA}"
}

github_api() {
  method="$1"
  url="$2"
  payload_file="$3"
  response_file="$(mktemp)"

  if curl -sS \
    -o "${response_file}" \
    -w "%{http_code}" \
    -X "${method}" \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${token}" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    -d "@${payload_file}" \
    "${url}" > "${response_file}.status"; then
    http_status="$(cat "${response_file}.status")"
  else
    curl_status=$?
    echo "GitHub API request failed before receiving an HTTP response: curl exit ${curl_status}" >&2
    echo "Request payload:" >&2
    cat "${payload_file}" >&2
    rm -f "${response_file}" "${response_file}.status"
    return "${curl_status}"
  fi

  if [ "${http_status}" -lt 200 ] || [ "${http_status}" -ge 300 ]; then
    echo "GitHub API request failed: ${method} ${url} returned HTTP ${http_status}" >&2
    echo "Response body:" >&2
    cat "${response_file}" >&2
    echo >&2
    echo "Request payload:" >&2
    cat "${payload_file}" >&2
    echo >&2
    rm -f "${response_file}" "${response_file}.status"
    return 1
  fi

  cat "${response_file}"
  rm -f "${response_file}" "${response_file}.status"
}

write_payload() {
  payload_file="$1"
  payload_kind="$2"
  python3 .github/scripts/github-check-run-payload.py \
    "$action" \
    "$payload_kind" \
    "$check_name" \
    "$GIT_SHA" \
    "${JOB_NAME:-}" \
    "${GITHUB_RUN_URL:-}" \
    > "${payload_file}"
}

payload_file="$(mktemp)"
trap 'rm -f "${payload_file}"' EXIT

if [ "${action}" = "started" ]; then
  if [ -n "${GITHUB_CHECK_RUN_ID:-}" ]; then
    printf '%s\n' "${GITHUB_CHECK_RUN_ID}" > "${check_id_file}"
    write_payload "${payload_file}" update
    github_api PATCH "${api_url}/${GITHUB_CHECK_RUN_ID}" "${payload_file}" >/dev/null
  else
    commit_status="$(github_get_commit)"
    if [ "${commit_status}" != "200" ]; then
      echo "Commit ${GIT_SHA} was not found in ${GITHUB_REPOSITORY}; GitHub returned HTTP ${commit_status}." >&2
      echo "The Checks API can only create a check run for a commit visible in the target repository." >&2
    fi
    write_payload "${payload_file}" create
    response="$(github_api POST "${api_url}" "${payload_file}")"
    printf '%s' "${response}" | python3 -c 'import json, sys; print(json.load(sys.stdin)["id"])' > "${check_id_file}"
  fi
  exit 0
fi

if [ -n "${GITHUB_CHECK_RUN_ID:-}" ]; then
  check_id="${GITHUB_CHECK_RUN_ID}"
  write_payload "${payload_file}" update
  github_api PATCH "${api_url}/${check_id}" "${payload_file}" >/dev/null
elif [ -s "${check_id_file}" ]; then
  check_id="$(cat "${check_id_file}")"
  write_payload "${payload_file}" update
  github_api PATCH "${api_url}/${check_id}" "${payload_file}" >/dev/null
else
  write_payload "${payload_file}" create
  github_api POST "${api_url}" "${payload_file}" >/dev/null
fi
