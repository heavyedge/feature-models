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

. .github/scripts/github-app.sh

doc_branch="${TAG_NAME}-doc"
doc_repo="/tmp/heavyedge-doc-repo-$$"
remote_url="https://github.com/${GITHUB_REPOSITORY}.git"

if ! git check-ref-format "refs/heads/${doc_branch}"; then
  echo "Invalid documentation branch name: ${doc_branch}" >&2
  exit 1
fi

auth_header="$(github_app_create_basic_auth_header '{"contents":"write"}')"

git init "${doc_repo}"
git -C "${doc_repo}" remote add origin "${remote_url}"
git -C "${doc_repo}" \
  -c "http.https://github.com/.extraheader=AUTHORIZATION: basic ${auth_header}" \
  fetch --depth=1 origin "refs/tags/${TAG_NAME}:refs/tags/${TAG_NAME}"
base_commit="$(git -C "${doc_repo}" rev-parse "FETCH_HEAD^{commit}")"
git -C "${doc_repo}" checkout -B "${doc_branch}" "${base_commit}"

git -C "${doc_repo}" rm -r --cached --ignore-unmatch .github/workflows/

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
