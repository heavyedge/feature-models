#!/bin/sh

set -eu

state_dir="${HEAVYEDGE_STATE_DIR:-/var/run/heavyedge}"
done_file="${POSTFIX_DONE_FILE:-${state_dir}/deploy-complete}"
result_file="${GITHUB_CHECK_RUN_RESULT_FILE:-${state_dir}/github-check-run-result}"

sh .github/scripts/github-check-run.sh started || true

while [ ! -f "${done_file}" ]; do
  sleep 5
done

if [ -r "${result_file}" ]; then
  result="$(cat "${result_file}")"
else
  result=failed
fi

case "${result}" in
  succeeded|failed)
    sh .github/scripts/github-check-run.sh "${result}" || true
    ;;
  *)
    echo "Unknown GitHub check run result '${result}'; reporting failure." >&2
    sh .github/scripts/github-check-run.sh failed || true
    ;;
esac
