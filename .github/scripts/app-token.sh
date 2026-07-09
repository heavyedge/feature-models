#!/bin/sh

set -eu

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

private_key_file="$(mktemp)"
trap 'rm -f "${private_key_file}"' EXIT
printf '%s\n' "${GH_APP_PRIVATE_KEY}" > "${private_key_file}"

base64url() {
  openssl base64 -A | tr '+/' '-_' | tr -d '='
}

now="$(date +%s)"
iat="$((now - 60))"
exp="$((now + 540))"

header="$(printf '{"alg":"RS256","typ":"JWT"}' | base64url)"
payload="$(printf '{"iat":%s,"exp":%s,"iss":"%s"}' "${iat}" "${exp}" "${GH_APP_ID}" | base64url)"
signature="$(
  printf '%s.%s' "${header}" "${payload}" \
    | openssl dgst -sha256 -sign "${private_key_file}" -binary \
    | base64url
)"
jwt="${header}.${payload}.${signature}"

installation_id="${GH_APP_INSTALLATION_ID:-}"
if [ -z "${installation_id}" ]; then
  installation_response="$(
    curl -fsS \
      -H "Accept: application/vnd.github+json" \
      -H "Authorization: Bearer ${jwt}" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/${GITHUB_REPOSITORY}/installation"
  )"
  installation_id="$(
    printf '%s' "${installation_response}" \
      | python -c 'import json,sys; print(json.load(sys.stdin)["id"])'
  )"
fi

token_response="$(
  curl -fsS \
    -X POST \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${jwt}" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/app/installations/${installation_id}/access_tokens" \
    -d '{"permissions":{"contents":"write"}}'
)"

printf '%s' "${token_response}" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["token"])'
