#!/bin/sh

set -eu

state_dir="${HEAVYEDGE_STATE_DIR:-/var/run/heavyedge}"
result_file="${DEPLOY_RESULT_FILE:-${state_dir}/deploy-result}"
notify_done_file="${NOTIFY_DEPLOY_DONE_FILE:-${state_dir}/notify-deploy-complete}"

finish_notify() {
  mkdir -p "$(dirname "${notify_done_file}")"
  touch "${notify_done_file}"
}

trap finish_notify EXIT

sh .github/scripts/notify-deploy.sh started || true

while [ ! -f "${result_file}" ]; do
  sleep 5
done

. "${result_file}"

case "${status:-}" in
  succeeded|failed)
    sh .github/scripts/notify-deploy.sh "${status}" || true
    ;;
  *)
    echo "Unknown deploy result status '${status:-}'; sending failure notification." >&2
    sh .github/scripts/notify-deploy.sh failed || true
    ;;
esac
