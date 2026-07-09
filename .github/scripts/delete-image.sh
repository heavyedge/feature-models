#!/bin/sh

set -eu

if [ -z "${DOCKER_REGISTRY:-}" ] ||
  [ -z "${DOCKER_NAMESPACE:-}" ] ||
  [ -z "${IMAGE_NAME:-}" ] ||
  [ -z "${IMAGE_TAG:-}" ]; then
  echo "Docker registry cleanup environment is incomplete; skipping ${IMAGE_TAG:-unknown} deletion." >&2
  exit 0
fi

if [ -z "${DOCKER_USERNAME:-}" ] || [ -z "${DOCKER_PASSWORD:-}" ]; then
  echo "Docker registry cleanup credentials are incomplete; skipping ${IMAGE_TAG} deletion." >&2
  exit 0
fi

docker_repository="${DOCKER_NAMESPACE}/${IMAGE_NAME}"

case "${DOCKER_REGISTRY}" in
  http://* | https://*)
    registry_url="${DOCKER_REGISTRY%/}"
    ;;
  *)
    registry_url="https://${DOCKER_REGISTRY%/}"
    ;;
esac

manifest_url="${registry_url}/v2/${docker_repository}/manifests/${IMAGE_TAG}"
headers_file="$(mktemp)"
trap 'rm -f "${headers_file}"' EXIT

accept_header="application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"

http_status="$(
  curl -sS \
    -o /dev/null \
    -w "%{http_code}" \
    -D "${headers_file}" \
    -u "${DOCKER_USERNAME}:${DOCKER_PASSWORD}" \
    -H "Accept: ${accept_header}" \
    -X HEAD \
    "${manifest_url}" || true
)"

if [ "${http_status}" = "404" ]; then
  echo "Docker image tag ${docker_repository}:${IMAGE_TAG} is already absent."
  exit 0
fi

if [ "${http_status}" -lt 200 ] || [ "${http_status}" -ge 300 ]; then
  echo "Could not inspect Docker image tag ${docker_repository}:${IMAGE_TAG}; registry returned HTTP ${http_status}." >&2
  exit 1
fi

digest="$(
  awk 'tolower($0) ~ /^docker-content-digest:/ { print $2; exit }' "${headers_file}" | tr -d '\r'
)"

if [ -z "${digest}" ]; then
  echo "Registry did not return Docker-Content-Digest for ${docker_repository}:${IMAGE_TAG}." >&2
  exit 1
fi

delete_url="${registry_url}/v2/${docker_repository}/manifests/${digest}"
delete_status="$(
  curl -sS \
    -o /dev/null \
    -w "%{http_code}" \
    -u "${DOCKER_USERNAME}:${DOCKER_PASSWORD}" \
    -X DELETE \
    "${delete_url}" || true
)"

if [ "${delete_status}" = "404" ] || [ "${delete_status}" = "202" ]; then
  echo "Deleted Docker image tag ${docker_repository}:${IMAGE_TAG}."
  exit 0
fi

echo "Could not delete Docker image tag ${docker_repository}:${IMAGE_TAG}; registry returned HTTP ${delete_status}." >&2
exit 1
