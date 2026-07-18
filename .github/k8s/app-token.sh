#!/bin/sh

set -eu

github_api() {
  response_file="$1"
  shift

  http_status=
  http_status="$(
    curl -sS \
      -o "${response_file}" \
      -w "%{http_code}" \
      "$@"
  )" || return 1

  [ "${http_status}" -ge 200 ] && [ "${http_status}" -lt 300 ]
}

report_github_api_error() {
  action="$1"
  status="${http_status:-curl-failed}"

  echo "::error::${action}; GitHub API returned HTTP ${status}." >&2
  if [ -s "${response_file}" ]; then
    sed 's/^/GitHub API response: /' "${response_file}" >&2
  fi
}

required_vars="
GH_APP_ID
GH_APP_PRIVATE_KEY
GITHUB_REPOSITORY
"

for var_name in ${required_vars}; do
  eval "var_value=\${${var_name}:-}"
  if [ -z "${var_value}" ]; then
    echo "::error::Missing ${var_name} for GitHub App token creation." >&2
    exit 1
  fi
done

if ! private_key_file="$(mktemp)"; then
  echo "::error::Failed to create temporary private key file." >&2
  exit 2
fi
trap 'rm -f "${private_key_file}"' EXIT
if ! printf '%s\n' "${GH_APP_PRIVATE_KEY}" > "${private_key_file}"; then
  echo "::error::Failed to write GitHub App private key." >&2
  exit 2
fi

base64url() {
  openssl base64 -A | tr '+/' '-_' | tr -d '='
}

now="$(date +%s)"
iat="$((now - 60))"
exp="$((now + 540))"

if ! header="$(printf '{"alg":"RS256","typ":"JWT"}' | base64url)"; then
  echo "::error::Failed to encode GitHub App JWT header." >&2
  exit 3
fi
if ! payload="$(printf '{"iat":%s,"exp":%s,"iss":"%s"}' "${iat}" "${exp}" "${GH_APP_ID}" | base64url)"; then
  echo "::error::Failed to encode GitHub App JWT payload." >&2
  exit 3
fi
if ! signature="$(
  printf '%s.%s' "${header}" "${payload}" \
    | openssl dgst -sha256 -sign "${private_key_file}" -binary \
    | base64url
)"; then
  echo "::error::Failed to sign GitHub App JWT." >&2
  exit 3
fi
jwt="${header}.${payload}.${signature}"

installation_id="${GH_APP_INSTALLATION_ID:-}"
if [ -z "${installation_id}" ]; then
  response_file="$(mktemp)"
  if ! github_api "${response_file}" \
      -H "Accept: application/vnd.github+json" \
      -H "Authorization: Bearer ${jwt}" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/${GITHUB_REPOSITORY}/installation"; then
    report_github_api_error "Failed to fetch GitHub App installation for ${GITHUB_REPOSITORY}"
    rm -f "${response_file}"
    exit 4
  fi
  if ! installation_id="$(
    python -c 'import json,sys; print(json.load(sys.stdin)["id"])' < "${response_file}"
  )"; then
    echo "::error::Failed to parse GitHub App installation ID." >&2
    rm -f "${response_file}"
    exit 5
  fi
  rm -f "${response_file}"
fi

response_file="$(mktemp)"
if ! github_api "${response_file}" \
    -X POST \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${jwt}" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/app/installations/${installation_id}/access_tokens" \
    -d '{"permissions":{"actions":"write","contents":"write"}}'; then
  report_github_api_error "Failed to create GitHub App installation token for installation ${installation_id}"
  rm -f "${response_file}"
  exit 6
fi

if ! python -c 'import json,sys; print(json.load(sys.stdin)["token"])' < "${response_file}"; then
  echo "::error::Failed to parse GitHub App installation token." >&2
  rm -f "${response_file}"
  exit 7
fi
rm -f "${response_file}"
