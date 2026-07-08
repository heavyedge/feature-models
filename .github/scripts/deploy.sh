#!/bin/sh
set -e

deploy_result_file="${DEPLOY_RESULT_FILE:-/var/run/heavyedge/deploy-result}"

finish_deploy() {
  exit_code=$?
  mkdir -p "$(dirname "${deploy_result_file}")"

  if [ "${exit_code}" -eq 0 ]; then
    result_status=succeeded
  else
    result_status=failed
  fi

  result_file_tmp="${deploy_result_file}.$$"
  {
    printf 'status=%s\n' "${result_status}"
    printf 'exit_code=%s\n' "${exit_code}"
  } > "${result_file_tmp}"
  mv "${result_file_tmp}" "${deploy_result_file}"
  exit "${exit_code}"
}

trap finish_deploy EXIT

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
