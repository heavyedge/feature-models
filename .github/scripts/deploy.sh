#!/bin/sh
set -e

write_check_status() {
  if [ -z "${DEPLOY_CHECKS_TMP_FILE:-}" ]; then
    return 0
  fi

  CHECK_KEY="$1" \
    CHECK_STATUS="$2" \
    CHECK_SUMMARY="$3" \
    python3 <<'PY'
import json
import os
import tempfile

path = os.environ["DEPLOY_CHECKS_TMP_FILE"]
key = os.environ["CHECK_KEY"]
status = os.environ["CHECK_STATUS"]
summary = os.environ["CHECK_SUMMARY"]
names = {
    "gpu_build": "GPU build",
    "upload_model": "Upload model",
    "upload_doc": "Upload document",
}

checks = {}
if os.path.exists(path):
    with open(path, encoding="utf-8") as file:
        parsed = json.load(file)
    if isinstance(parsed, dict):
        checks = parsed

checks[key] = {
    "name": names.get(key, key),
    "status": status,
    "summary": summary,
}

directory = os.path.dirname(path) or "."
os.makedirs(directory, exist_ok=True)
with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as file:
    json.dump(checks, file, sort_keys=True)
    file.write("\n")
    tmp_path = file.name
os.replace(tmp_path, path)
PY
}

fail_unfinished_checks() {
  if [ -z "${DEPLOY_CHECKS_TMP_FILE:-}" ]; then
    return 0
  fi

  python3 <<'PY'
import json
import os
import tempfile

path = os.environ["DEPLOY_CHECKS_TMP_FILE"]
if not os.path.exists(path):
    raise SystemExit

with open(path, encoding="utf-8") as file:
    checks = json.load(file)
if not isinstance(checks, dict):
    raise SystemExit

for check in checks.values():
    if not isinstance(check, dict):
        continue
    if check.get("status") in {"pending", "in_progress"}:
        check["status"] = "failed"
        check["summary"] = "Deployment stopped before this step completed."

directory = os.path.dirname(path) or "."
with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as file:
    json.dump(checks, file, sort_keys=True)
    file.write("\n")
    tmp_path = file.name
os.replace(tmp_path, path)
PY
}

run_phase() {
  phase_key="$1"
  phase_name="$2"
  shift 2

  write_check_status "${phase_key}" in_progress "${phase_name} is running."
  set +e
  (
    set -e
    "$@"
  )
  phase_exit_code=$?
  set -e

  if [ "${phase_exit_code}" -eq 0 ]; then
    write_check_status "${phase_key}" succeeded "${phase_name} succeeded."
  else
    write_check_status "${phase_key}" failed "${phase_name} failed."
    fail_unfinished_checks
    exit "${phase_exit_code}"
  fi
}

gpu_build() {
  uv pip install --system -r requirements.txt -r notebooks/requirements.txt

  HEAVYEDGE_GPU_DEVICES=$(python3 scripts/cuda-preflight.py --print-devices)
  export HEAVYEDGE_GPU_DEVICES
  if [ -z "${MAKE_JOBS:-}" ] || [ "${MAKE_JOBS}" = "auto" ]; then
    MAKE_JOBS=$(python3 scripts/cuda-preflight.py --print-count)
  fi
  export MAKE_JOBS
  python3 scripts/cuda-preflight.py

  HEAVYEDGE_TEST_MODE=${HEAVYEDGE_TEST_MODE} make -j "${MAKE_JOBS}" models notebooks
}

upload_model() {
  if [ -z "${TAG_NAME:-}" ]; then
    echo "TAG_NAME is required to upload to Hugging Face" >&2
    exit 1
  fi
  uv pip install --system huggingface_hub
  if [ -n "${DEPLOY_METADATA_TMP_FILE:-}" ]; then
    python upload.py "${TAG_NAME}" --metadata-file "${DEPLOY_METADATA_TMP_FILE}"
  else
    python upload.py "${TAG_NAME}"
  fi
}

upload_doc() {
  sh .github/scripts/push-doc.sh
}

write_check_status gpu_build pending "GPU build is waiting."
if [ "${UPLOAD_TO_HUGGINGFACE}" = "1" ]; then
  write_check_status upload_model pending "Model upload is waiting."
else
  write_check_status upload_model skipped "Model upload is disabled."
fi
if [ "${PUSH_DOC}" = "1" ]; then
  write_check_status upload_doc pending "Document upload is waiting."
else
  write_check_status upload_doc skipped "Document upload is disabled."
fi

run_phase gpu_build "GPU build" gpu_build

if [ "${UPLOAD_TO_HUGGINGFACE}" = "1" ]; then
  run_phase upload_model "Upload model" upload_model
fi

if [ "${PUSH_DOC}" = "1" ]; then
  run_phase upload_doc "Upload document" upload_doc
fi
