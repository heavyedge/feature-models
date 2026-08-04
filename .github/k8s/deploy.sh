#!/bin/sh

set -eu

deploy_status=0

if [ "${DEPLOY_MODE:-false}" = "true" ]; then
  if [ -z "${GITHUB_REF_NAME:-}" ] || [ -z "${HUGGINGFACE_TOKEN:-}" ]; then
    echo "::error::GITHUB_REF_NAME and HUGGINGFACE_TOKEN are required." >&2
    deploy_status=$((deploy_status | 2))
  elif ! uv pip install --system huggingface_hub; then
    deploy_status=$((deploy_status | 2))
  elif ! python upload.py "${GITHUB_REF_NAME}"; then
    deploy_status=$((deploy_status | 2))
  fi
else
  echo "Skipping huggingface upload."
fi

if [ "${DOC_DEPLOY_MODE:-false}" = "true" ]; then
  if [ -z "${GITHUB_REPOSITORY:-}" ] || [ -z "${GITHUB_REF_NAME:-}" ] || [ -z "${GITHUB_APP_TOKEN:-}" ]; then
    echo "::error::GITHUB_REPOSITORY, GITHUB_REF_NAME, and GITHUB_APP_TOKEN are required for release asset upload." >&2
    deploy_status=$((deploy_status | 4))
  elif [ ! -d benchmarks ] || [ ! -d examples ]; then
    echo "::error::Build output directories benchmarks and examples are required." >&2
    deploy_status=$((deploy_status | 4))
  else
    archive_file="$(mktemp)"
    response_file="$(mktemp)"
    trap 'rm -f "$archive_file" "$response_file"' EXIT INT TERM
    asset_name="results-${GITHUB_REF_NAME}.tar.gz"
    release_url="https://api.github.com/repos/${GITHUB_REPOSITORY}/releases/tags/${GITHUB_REF_NAME}"

    if ! tar -czf "$archive_file" benchmarks examples; then
      deploy_status=$((deploy_status | 4))
    elif ! http_status="$(curl -sS -o "$response_file" -w '%{http_code}' \
        -H 'Accept: application/vnd.github+json' \
        -H "Authorization: Bearer ${GITHUB_APP_TOKEN}" \
        -H 'X-GitHub-Api-Version: 2022-11-28' "$release_url")" || [ "$http_status" -lt 200 ] || [ "$http_status" -ge 300 ]; then
      echo "::error::Failed to find release ${GITHUB_REF_NAME}; GitHub API returned HTTP ${http_status:-curl-failed}." >&2
      deploy_status=$((deploy_status | 4))
    else
      upload_url="$(python -c 'import json, sys; print(json.load(sys.stdin)["upload_url"].split("{")[0])' < "$response_file")"
      release_asset_id="$(python -c 'import json, sys; name = sys.argv[1]; print(next((asset["id"] for asset in json.load(sys.stdin)["assets"] if asset["name"] == name), ""))' "$asset_name" < "$response_file")"

      if [ -n "$release_asset_id" ]; then
        curl -sS -o "$response_file" -X DELETE \
          -H 'Accept: application/vnd.github+json' \
          -H "Authorization: Bearer ${GITHUB_APP_TOKEN}" \
          -H 'X-GitHub-Api-Version: 2022-11-28' \
          "https://api.github.com/repos/${GITHUB_REPOSITORY}/releases/assets/${release_asset_id}"
      fi

      if ! http_status="$(curl -sS -o "$response_file" -w '%{http_code}' -X POST \
          -H 'Accept: application/vnd.github+json' \
          -H "Authorization: Bearer ${GITHUB_APP_TOKEN}" \
          -H 'Content-Type: application/gzip' \
          -H 'X-GitHub-Api-Version: 2022-11-28' \
          --data-binary "@${archive_file}" "${upload_url}?name=${asset_name}")" || [ "$http_status" -lt 200 ] || [ "$http_status" -ge 300 ]; then
        echo "::error::Failed to upload ${asset_name}; GitHub API returned HTTP ${http_status:-curl-failed}." >&2
        deploy_status=$((deploy_status | 4))
      else
        echo "Uploaded ${asset_name} to release ${GITHUB_REF_NAME}."
      fi
    fi
  fi
else
  echo "Skipping benchmarks and examples upload."
fi

exit "$deploy_status"
