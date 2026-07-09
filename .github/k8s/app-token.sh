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
  if ! installation_response="$(
    curl -fsS \
      -H "Accept: application/vnd.github+json" \
      -H "Authorization: Bearer ${jwt}" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/${GITHUB_REPOSITORY}/installation"
  )"; then
    echo "::error::Failed to fetch GitHub App installation." >&2
    exit 4
  fi
  if ! installation_id="$(
    printf '%s' "${installation_response}" \
      | python -c 'import json,sys; print(json.load(sys.stdin)["id"])'
  )"; then
    echo "::error::Failed to parse GitHub App installation ID." >&2
    exit 5
  fi
fi

if ! token_response="$(
  curl -fsS \
    -X POST \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${jwt}" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/app/installations/${installation_id}/access_tokens" \
    -d '{"permissions":{"actions":"write","contents":"write"}}'
)"; then
  echo "::error::Failed to create GitHub App installation token." >&2
  exit 6
fi

if ! printf '%s' "${token_response}" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["token"])'; then
  echo "::error::Failed to parse GitHub App installation token." >&2
  exit 7
fi
