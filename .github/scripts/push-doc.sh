#!/bin/sh
set -e

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
state_dir="${HEAVYEDGE_STATE_DIR:-/var/run/heavyedge}"
token_file="${GITHUB_APP_TOKEN_FILE:-${state_dir}/github-app-token}"
token_error_file="${GITHUB_APP_TOKEN_ERROR_FILE:-${state_dir}/github-app-token-error}"
token_wait_seconds="${GITHUB_APP_TOKEN_WAIT_SECONDS:-120}"

if ! git check-ref-format "refs/heads/${doc_branch}"; then
  echo "Invalid documentation branch name: ${doc_branch}" >&2
  exit 1
fi

wait_for_github_token() {
  waited=0
  while [ ! -s "${token_file}" ]; do
    if [ -s "${token_error_file}" ]; then
      echo "GitHub App token sidecar reported an error; cannot push documentation branch." >&2
      cat "${token_error_file}" >&2
      return 1
    fi
    if [ "${waited}" -ge "${token_wait_seconds}" ]; then
      echo "Timed out waiting for GitHub App token at ${token_file}; cannot push documentation branch." >&2
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
}

wait_for_github_token
github_app_token="$(cat "${token_file}")"
auth_header="$(printf 'x-access-token:%s' "${github_app_token}" | base64 | tr -d '\n')"

git init "${doc_repo}"
git -C "${doc_repo}" remote add origin "${remote_url}"
git -C "${doc_repo}" \
  -c "http.https://github.com/.extraheader=AUTHORIZATION: basic ${auth_header}" \
  fetch --depth=1 origin "refs/tags/${TAG_NAME}:refs/tags/${TAG_NAME}"
base_commit="$(git -C "${doc_repo}" rev-parse "FETCH_HEAD^{commit}")"
git -C "${doc_repo}" checkout -B "${doc_branch}" "${base_commit}"

cp notebooks/*.ipynb "${doc_repo}/notebooks/"

git -C "${doc_repo}" config user.name "${GIT_AUTHOR_NAME}"
git -C "${doc_repo}" config user.email "${GIT_AUTHOR_EMAIL}"
git -C "${doc_repo}" \
  -c filter.nbstripout.clean=cat \
  -c filter.nbstripout.smudge=cat \
  -c filter.nbstripout.required=false \
  add notebooks

if git -C "${doc_repo}" diff --cached --quiet; then
  echo "No changes to commit for ${doc_branch}"
else
  git -C "${doc_repo}" commit -m "Build notebooks for ${TAG_NAME}"
fi

git -C "${doc_repo}" \
  -c "http.https://github.com/.extraheader=AUTHORIZATION: basic ${auth_header}" \
  push --force origin "HEAD:refs/heads/${doc_branch}"
