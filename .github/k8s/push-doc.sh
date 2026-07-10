#!/bin/sh
set -e

if [ -z "${GITHUB_REPOSITORY}" ]; then
  echo "Missing GITHUB_REPOSITORY for pushing built notebooks" >&2
  exit 1
fi
if [ -z "${GITHUB_APP_TOKEN}" ]; then
  echo "Missing GITHUB_APP_TOKEN for pushing built notebooks" >&2
  exit 1
fi

if [ "${PUSH_DOC:-0}" = "1" ]; then
  doc_branch="${DOC_BRANCH:-${GITHUB_SHA}-doc}"
else
  doc_branch="${DOC_BRANCH:-${GITHUB_SHA}-doc}-test"
fi
fetch_ref="${GITHUB_SHA}"
doc_repo="/tmp/heavyedge-doc-repo-$$"
remote_url="https://github.com/${GITHUB_REPOSITORY}.git"
git_author_name="${GIT_AUTHOR_NAME:-heavyedge-bot}"
git_author_email="${GIT_AUTHOR_EMAIL:-heavyedge-bot@users.noreply.github.com}"

if ! git check-ref-format "refs/heads/${doc_branch}"; then
  echo "Invalid documentation branch name: ${doc_branch}" >&2
  exit 1
fi

auth_header="$(printf 'x-access-token:%s' "${GITHUB_APP_TOKEN}" | openssl base64 -A)"

git init "${doc_repo}"
git -C "${doc_repo}" remote add origin "${remote_url}"
git -C "${doc_repo}" \
  -c "http.https://github.com/.extraheader=AUTHORIZATION: basic ${auth_header}" \
  fetch --depth=1 origin "${fetch_ref}"
base_commit="$(git -C "${doc_repo}" rev-parse "FETCH_HEAD^{commit}")"
git -C "${doc_repo}" checkout -B "${doc_branch}" "${base_commit}"

cp notebooks/*.ipynb "${doc_repo}/notebooks/"

git -C "${doc_repo}" config user.name "${git_author_name}"
git -C "${doc_repo}" config user.email "${git_author_email}"
git -C "${doc_repo}" \
  -c filter.nbstripout.clean=cat \
  -c filter.nbstripout.smudge=cat \
  -c filter.nbstripout.required=false \
  add notebooks

if git -C "${doc_repo}" diff --cached --quiet; then
  echo "No changes to commit for ${doc_branch}"
else
  git -C "${doc_repo}" commit -m "Build doc for ${doc_branch%-doc}"
fi

git -C "${doc_repo}" \
  -c "http.https://github.com/.extraheader=AUTHORIZATION: basic ${auth_header}" \
  push --force origin "HEAD:refs/heads/${doc_branch}"

if [ ! "${PUSH_DOC:-0}" = "1" ]; then
  git -C "${doc_repo}" \
    -c "http.https://github.com/.extraheader=AUTHORIZATION: basic ${auth_header}" \
    push origin --delete "${doc_branch}"
fi
