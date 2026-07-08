#!/bin/sh

github_app_base64_urlencode() {
  base64 | tr '+/' '-_' | tr -d '=\n'
}

github_app_require_env() {
  if [ -z "${GH_APP_ID:-}" ]; then
    echo "Missing GH_APP_ID secret for GitHub App authentication." >&2
    return 1
  fi
  if [ -z "${GH_APP_PRIVATE_KEY:-}" ]; then
    echo "Missing GH_APP_PRIVATE_KEY secret for GitHub App authentication." >&2
    return 1
  fi
  if [ -z "${GITHUB_REPOSITORY:-}" ]; then
    echo "Missing GITHUB_REPOSITORY for GitHub App authentication." >&2
    return 1
  fi
}

github_app_create_jwt() {
  github_app_require_env

  private_key_file="$(mktemp)"
  printf '%b\n' "${GH_APP_PRIVATE_KEY}" > "${private_key_file}"

  now="$(date +%s)"
  iat=$((now - 60))
  exp=$((now + 540))
  jwt_header="$(printf '{"alg":"RS256","typ":"JWT"}' | github_app_base64_urlencode)"
  jwt_payload="$(printf '{"iat":%s,"exp":%s,"iss":"%s"}' "${iat}" "${exp}" "${GH_APP_ID}" | github_app_base64_urlencode)"
  jwt_signature="$(printf '%s.%s' "${jwt_header}" "${jwt_payload}" | openssl dgst -sha256 -sign "${private_key_file}" -binary | github_app_base64_urlencode)"

  rm -f "${private_key_file}"
  printf '%s' "${jwt_header}.${jwt_payload}.${jwt_signature}"
}

github_app_get_installation_id() {
  if [ -n "${GH_APP_INSTALLATION_ID:-}" ]; then
    printf '%s' "${GH_APP_INSTALLATION_ID}"
    return 0
  fi

  github_app_jwt="${1:-$(github_app_create_jwt)}"
  owner="${GITHUB_REPOSITORY%/*}"
  repo="${GITHUB_REPOSITORY#*/}"
  installation_response="$(curl -fsS \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${github_app_jwt}" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/${owner}/${repo}/installation")"

  printf '%s' "${installation_response}" | python -c 'import json, sys; print(json.load(sys.stdin)["id"])'
}

github_app_api() {
  method="$1"
  url="$2"
  body="${3:-}"
  response_file="$(mktemp)"
  status_file="$(mktemp)"

  if [ -n "${body}" ]; then
    curl -sS \
      -o "${response_file}" \
      -w "%{http_code}" \
      -X "${method}" \
      -H "Accept: application/vnd.github+json" \
      -H "Authorization: Bearer ${github_app_jwt}" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      -d "${body}" \
      "${url}" > "${status_file}"
  else
    curl -sS \
      -o "${response_file}" \
      -w "%{http_code}" \
      -X "${method}" \
      -H "Accept: application/vnd.github+json" \
      -H "Authorization: Bearer ${github_app_jwt}" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "${url}" > "${status_file}"
  fi

  curl_status=$?
  if [ "${curl_status}" -ne 0 ]; then
    echo "GitHub App API request failed before receiving an HTTP response: curl exit ${curl_status}" >&2
    rm -f "${response_file}" "${status_file}"
    return "${curl_status}"
  fi

  http_status="$(cat "${status_file}")"
  if [ "${http_status}" -lt 200 ] || [ "${http_status}" -ge 300 ]; then
    echo "GitHub App API request failed: ${method} ${url} returned HTTP ${http_status}" >&2
    echo "Response body:" >&2
    cat "${response_file}" >&2
    echo >&2
    if [ -n "${body}" ]; then
      echo "Request body:" >&2
      printf '%s\n' "${body}" >&2
    fi
    rm -f "${response_file}" "${status_file}"
    return 1
  fi

  cat "${response_file}"
  rm -f "${response_file}" "${status_file}"
}

github_app_create_installation_token() {
  if [ "$#" -gt 0 ]; then
    permissions_json="$1"
  else
    permissions_json='{"contents":"write"}'
  fi

  github_app_jwt="$(github_app_create_jwt)"
  installation_id="$(github_app_get_installation_id "${github_app_jwt}")"
  token_request_body="$(printf '{"permissions":%s}' "${permissions_json}")"
  token_response="$(github_app_api \
    POST \
    "https://api.github.com/app/installations/${installation_id}/access_tokens" \
    "${token_request_body}")"

  printf '%s' "${token_response}" | python -c 'import json, sys; print(json.load(sys.stdin)["token"])'
}

github_app_create_basic_auth_header() {
  github_app_token="$(github_app_create_installation_token "$@")"
  printf 'x-access-token:%s' "${github_app_token}" | base64 | tr -d '\n'
}
