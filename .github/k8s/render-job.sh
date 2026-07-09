#!/bin/sh
set -eu

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
template_file="${1:-${script_dir}/job.yaml}"
containers_dir="${2:-${script_dir}/containers}"
tmp_file="$(mktemp "${TMPDIR:-/tmp}/job.XXXXXX")"
container_files='
build.yaml
watch-build.yaml
image-cleanup.yaml
email-notify.yaml
github-app-token.yaml
github-check.yaml
dispatch-workflow.yaml
postfix.yaml
'

cleanup() {
  rm -f "${tmp_file}"
}

trap cleanup EXIT

render_template() {
  inserted_containers=0

  while IFS= read -r line || [ -n "${line}" ]; do
    case "${line}" in
      "        # Container definitions are rendered from "*)
        continue
        ;;
    esac

    printf '%s\n' "${line}"

    if [ "${line}" = "      containers:" ]; then
      for container_file_name in ${container_files}; do
        container_file="${containers_dir}/${container_file_name}"
        if [ ! -f "${container_file}" ]; then
          echo "Missing container definition: ${container_file}" >&2
          return 1
        fi

        sed -n '1,$p' "${container_file}"
      done

      inserted_containers=1
    fi
  done < "${template_file}"

  if [ "${inserted_containers}" -eq 0 ]; then
    echo "Missing containers section in ${template_file}." >&2
    return 1
  fi
}

render_template > "${tmp_file}"

template_vars='$JOB_NAME $KUBE_NAMESPACE $KUBE_USERNAME $GIT_SHA $EVENT_NAME $MLXP_ZONE $DOCKER_REGISTRY $DOCKER_USERNAME $DOCKER_TAG $TEMP_DOCKER_TAG $DELETE_IMAGE_SCRIPT_BASE64 $CPU_REQUEST $GPU_REQUEST $GITHUB_REPOSITORY $GITHUB_RUN_ID $GITHUB_GPU_BUILD_CHECK_RUN_ID $GITHUB_UPLOAD_MODEL_CHECK_RUN_ID $GITHUB_UPLOAD_DOC_CHECK_RUN_ID $RELEASE_TAG_NAME $HEAVYEDGE_TEST_MODE $UPLOAD_TO_HUGGINGFACE $PUSH_DOC $GIT_AUTHOR_NAME $GIT_AUTHOR_EMAIL $SMTP_NOTIFY_TO $BUILD_FAILURE_SCENARIO'
envsubst "${template_vars}" < "${tmp_file}"
