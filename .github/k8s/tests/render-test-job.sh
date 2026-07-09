#!/bin/sh
set -eu

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
k8s_dir="$(CDPATH= cd -- "${script_dir}/.." && pwd)"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/heavyedge-k8s-test.XXXXXX")"

cleanup() {
  rm -rf "${tmp_dir}"
}

trap cleanup EXIT

cp "${k8s_dir}"/containers/*.yaml "${tmp_dir}/"
cp "${script_dir}/containers/build.yaml" "${tmp_dir}/build.yaml"

sh "${k8s_dir}/render-job.sh" "${k8s_dir}/job.yaml" "${tmp_dir}"
