#!/bin/sh
set -e

job_done_file="${POSTFIX_DONE_FILE:-/var/run/heavyedge/deploy-complete}"
github_check_run_result_file="${GITHUB_CHECK_RUN_RESULT_FILE:-/var/run/heavyedge/github-check-run-result}"

notify_deploy() {
  sh .github/scripts/notify-deploy.sh "$1" || true
}

finish_deploy() {
  status=$?
  mkdir -p "$(dirname "${job_done_file}")"

  if [ "${status}" -eq 0 ]; then
    printf '%s\n' succeeded > "${github_check_run_result_file}"
    notify_deploy succeeded
  else
    printf '%s\n' failed > "${github_check_run_result_file}"
    notify_deploy failed
  fi

  touch "${job_done_file}"
  exit "${status}"
}

trap finish_deploy EXIT

notify_deploy started

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
if [ "${PUSH_DOC}" = "1" ]; then
    uv pip install --system -r requirements.txt -r notebooks/requirements.txt
else
    uv pip install --system -r requirements.txt
fi

HEAVYEDGE_GPU_DEVICES=$(python3 scripts/cuda-preflight.py --print-devices)
export HEAVYEDGE_GPU_DEVICES
if [ -z "${MAKE_JOBS:-}" ] || [ "${MAKE_JOBS}" = "auto" ]; then
  MAKE_JOBS=$(python3 scripts/cuda-preflight.py --print-count)
fi
export MAKE_JOBS
python3 scripts/cuda-preflight.py

if [ "${PUSH_DOC}" = "1" ]; then
    HEAVYEDGE_TEST_MODE=${HEAVYEDGE_TEST_MODE} make -j "${MAKE_JOBS}" models notebooks
else
    HEAVYEDGE_TEST_MODE=${HEAVYEDGE_TEST_MODE} make -j "${MAKE_JOBS}" models
fi

# Deploy model
if [ "${UPLOAD_TO_HUGGINGFACE}" = "1" ]; then
  uv pip install --system huggingface_hub
  python upload.py
fi

# Deploy notebooks
if [ "${PUSH_DOC}" = "1" ]; then
  sh .github/scripts/push-doc.sh
fi
