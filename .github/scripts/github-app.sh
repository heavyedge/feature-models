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

github_app_create_installation_token() {
  if [ "$#" -gt 0 ]; then
    permissions_json="$1"
  else
    permissions_json='{"contents":"write"}'
  fi

  github_app_jwt="$(github_app_create_jwt)"
  installation_id="$(github_app_get_installation_id "${github_app_jwt}")"
  token_response="$(curl -fsS \
    -X POST \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${github_app_jwt}" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    -d "$(printf '{"permissions":%s}' "${permissions_json}")" \
    "https://api.github.com/app/installations/${installation_id}/access_tokens")"

  printf '%s' "${token_response}" | python -c 'import json, sys; print(json.load(sys.stdin)["token"])'
}

github_app_create_basic_auth_header() {
  github_app_token="$(github_app_create_installation_token "$@")"
  printf 'x-access-token:%s' "${github_app_token}" | base64 | tr -d '\n'
}
