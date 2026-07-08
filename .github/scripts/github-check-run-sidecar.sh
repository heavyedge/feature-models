#!/bin/sh

set -eu

state_dir="${HEAVYEDGE_STATE_DIR:-/var/run/heavyedge}"
result_file="${DEPLOY_RESULT_FILE:-${state_dir}/deploy-result}"

sh .github/scripts/github-check-run.sh started || true

while [ ! -f "${result_file}" ]; do
  sleep 5
done

. "${result_file}"

case "${status:-}" in
  succeeded|failed)
    sh .github/scripts/github-check-run.sh "${status}" || true
    ;;
  *)
    echo "Unknown deploy result status '${status:-}'; reporting failure." >&2
    sh .github/scripts/github-check-run.sh failed || true
    ;;
esac
