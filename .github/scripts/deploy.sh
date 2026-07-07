#!/bin/sh
set -e

# Check revision
if [ ! -r /etc/heavyedge/image-revision ]; then
  echo "Missing image revision file: /etc/heavyedge/image-revision" >&2
  exit 1
fi
if [ "$(cat /etc/heavyedge/image-revision)" != "${GIT_SHA}" ]; then
  echo "Image revision mismatch: expected ${GIT_SHA}" >&2
  echo "Actual image revision:" >&2
  cat /etc/heavyedge/image-revision >&2
  exit 1
fi

# Build model
pip install -r requirements.txt
HEAVYEDGE_TEST_MODE=${HEAVYEDGE_TEST_MODE} make models

# Deploy model
if [ "${UPLOAD_TO_HUGGINGFACE}" = "1" ]; then
  python upload.py
fi

# Build and push notebook
if [ "${PUSH_DOC}" = "1" ]; then
  HEAVYEDGE_TEST_MODE=${HEAVYEDGE_TEST_MODE} make notebooks

  if [ -z "${DOCS_GITHUB_TOKEN}" ]; then
    echo "Missing DOCS_GITHUB_TOKEN for pushing built notebooks" >&2
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
  auth_header="$(printf 'x-access-token:%s' "${DOCS_GITHUB_TOKEN}" | base64 | tr -d '\n')"

  if ! git check-ref-format "refs/heads/${doc_branch}"; then
    echo "Invalid documentation branch name: ${doc_branch}" >&2
    exit 1
  fi

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
  git -C "${doc_repo}" config filter.nbstripout.clean cat
  git -C "${doc_repo}" config filter.nbstripout.smudge cat
  git -C "${doc_repo}" config filter.nbstripout.required false
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
fi
