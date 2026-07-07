#!/bin/sh
set -e

if [ -z "${GH_APP_ID}" ]; then
  echo "Missing GH_APP_ID for pushing built notebooks" >&2
  exit 1
fi
if [ -z "${GH_APP_PRIVATE_KEY}" ]; then
  echo "Missing GH_APP_PRIVATE_KEY for pushing built notebooks" >&2
  exit 1
fi
if [ -z "${GITHUB_REPOSITORY}" ]; then
  echo "Missing GITHUB_REPOSITORY for pushing built notebooks" >&2
  exit 1
fi
if [ -z "${TAG_NAME}" ]; then
  echo "Missing TAG_NAME for pushing built notebooks" >&2
  exit 1
fi

doc_branch="${TAG_NAME}-doc"
doc_repo="/tmp/heavyedge-doc-repo-$$"
remote_url="https://github.com/${GITHUB_REPOSITORY}.git"

if ! git check-ref-format "refs/heads/${doc_branch}"; then
  echo "Invalid documentation branch name: ${doc_branch}" >&2
  exit 1
fi

private_key_file="$(mktemp)"
printf '%b\n' "${GH_APP_PRIVATE_KEY}" > "${private_key_file}"
trap 'rm -f "${private_key_file}"' EXIT

base64_urlencode() {
  base64 | tr '+/' '-_' | tr -d '=\n'
}

now="$(date +%s)"
iat=$((now - 60))
exp=$((now + 540))
jwt_header="$(printf '{"alg":"RS256","typ":"JWT"}' | base64_urlencode)"
jwt_payload="$(printf '{"iat":%s,"exp":%s,"iss":"%s"}' "${iat}" "${exp}" "${GH_APP_ID}" | base64_urlencode)"
jwt_signature="$(printf '%s.%s' "${jwt_header}" "${jwt_payload}" | openssl dgst -sha256 -sign "${private_key_file}" -binary | base64_urlencode)"
github_app_jwt="${jwt_header}.${jwt_payload}.${jwt_signature}"

if [ -z "${GH_APP_INSTALLATION_ID}" ]; then
  owner="${GITHUB_REPOSITORY%/*}"
  repo="${GITHUB_REPOSITORY#*/}"
  installation_response="$(curl -fsS \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${github_app_jwt}" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/${owner}/${repo}/installation")"
  installation_id="$(printf '%s' "${installation_response}" | python -c 'import json, sys; print(json.load(sys.stdin)["id"])')"
else
  installation_id="${GH_APP_INSTALLATION_ID}"
fi

token_response="$(curl -fsS \
  -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${github_app_jwt}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -d '{"permissions":{"contents":"write"}}' \
  "https://api.github.com/app/installations/${installation_id}/access_tokens")"
docs_github_token="$(printf '%s' "${token_response}" | python -c 'import json, sys; print(json.load(sys.stdin)["token"])')"
auth_header="$(printf 'x-access-token:%s' "${docs_github_token}" | base64 | tr -d '\n')"

git init "${doc_repo}"
git -C "${doc_repo}" remote add origin "${remote_url}"
git -C "${doc_repo}" \
  -c "http.https://github.com/.extraheader=AUTHORIZATION: basic ${auth_header}" \
  fetch --depth=1 origin "refs/tags/${TAG_NAME}:refs/tags/${TAG_NAME}"
base_commit="$(git -C "${doc_repo}" rev-parse "FETCH_HEAD^{commit}")"
git -C "${doc_repo}" checkout -B "${doc_branch}" "${base_commit}"

cp notebooks/*.ipynb "${doc_repo}/notebooks/"

git -C "${doc_repo}" config user.name "${DOCS_GIT_AUTHOR_NAME:-heavyedge-doc-bot}"
git -C "${doc_repo}" config user.email "${DOCS_GIT_AUTHOR_EMAIL:-heavyedge-doc-bot@users.noreply.github.com}"
git -C "${doc_repo}" \
  -c filter.nbstripout.clean=cat \
  -c filter.nbstripout.smudge=cat \
  -c filter.nbstripout.required=false \
  add notebooks

if git -C "${doc_repo}" diff --cached --quiet -- notebooks; then
  echo "No notebook changes to commit for ${doc_branch}"
else
  git -C "${doc_repo}" commit -m "Build notebooks for ${TAG_NAME}"
fi

git -C "${doc_repo}" \
  -c "http.https://github.com/.extraheader=AUTHORIZATION: basic ${auth_header}" \
  push --force origin "HEAD:refs/heads/${doc_branch}"
