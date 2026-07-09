#!/bin/sh

set -eu

require_env() {
  name="$1"
  eval "value=\${$name:-}"
  if [ -z "$value" ]; then
    echo "Missing required environment variable: $name" >&2
    exit 1
  fi
}

require_env DOCKER_REGISTRY
require_env DOCKER_USERNAME
require_env DOCKER_PASSWORD
require_env DOCKER_NAMESPACE
require_env IMAGE_NAME
require_env IMAGE_TAG

registry="${DOCKER_REGISTRY#http://}"
registry="${registry#https://}"
registry="${registry%/}"
repository="${DOCKER_NAMESPACE}/${IMAGE_NAME}"

headers_file="$(mktemp)"
cleanup() {
  rm -f "$headers_file"
}
trap cleanup EXIT

api_base="https://${registry}/v2/${repository}"
accept_header="Accept: application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json"

run_curl() {
  if curl "$@"; then
    return 0
  fi

  status="$?"
  if [ "$status" -eq 35 ]; then
    echo "curl failed with connection reset; retrying once." >&2
    sleep 3
    curl "$@"
    return "$?"
  fi

  return "$status"
}

echo "Resolving digest for ${registry}/${repository}:${IMAGE_TAG}"
run_curl --fail --silent --show-error --location \
  --request HEAD \
  --user "${DOCKER_USERNAME}:${DOCKER_PASSWORD}" \
  --header "$accept_header" \
  --dump-header "$headers_file" \
  --output /dev/null \
  "${api_base}/manifests/${IMAGE_TAG}"

digest="$(
  sed -n 's/^[Dd]ocker-[Cc]ontent-[Dd]igest:[[:space:]]*\([^[:space:]\r]*\).*/\1/p' "$headers_file" |
    tail -n 1
)"

if [ -z "$digest" ]; then
  echo "Docker-Content-Digest header was not returned for ${registry}/${repository}:${IMAGE_TAG}" >&2
  exit 1
fi

echo "Deleting ${registry}/${repository}@${digest}"
run_curl --fail --silent --show-error --location \
  --request DELETE \
  --user "${DOCKER_USERNAME}:${DOCKER_PASSWORD}" \
  "${api_base}/manifests/${digest}"

echo "Deleted ${registry}/${repository}:${IMAGE_TAG}"
