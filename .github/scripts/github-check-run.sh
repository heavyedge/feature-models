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

github_api() {
  method="$1"
  url="$2"
  payload_file="$3"

  curl --fail-with-body -sS \
    -X "${method}" \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${token}" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    -d "@${payload_file}" \
    "${url}"
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
