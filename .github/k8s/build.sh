#!/bin/sh

set -eu

if ! ./setup.sh; then
  exit 1
fi

make_targets="models"
case "${BUILD_MODE:-test}" in
  build)
    if ! HEAVYEDGE_TEST_MODE=0 make -j ${MAKE_JOBS} ${make_targets}; then
      exit 2
    fi
    ;;
  pull)
    overlay_dir="$(mktemp -d)"
    trap 'rm -rf "$overlay_dir"' EXIT INT TERM
    cp -a models/. "$overlay_dir/"
    if ! "$HOME/.local/bin/hf" download "${UPSTREAM_REPO_ID}" \
        --repo-type model \
        --revision "${UPSTREAM_REVISION}" \
        --token "${HUGGINGFACE_TOKEN}" \
        --local-dir models; then
      exit 2
    fi
    cp -a "$overlay_dir/." models/
    rm -rf models/.cache/huggingface
    ;;
  test)
    if ! HEAVYEDGE_TEST_MODE=1 make -j ${MAKE_JOBS} ${make_targets}; then
      exit 2
    fi
    ;;
  *)
    echo "::error::Unsupported build mode: ${BUILD_MODE}" >&2
    exit 2
    ;;
esac

if ! HEAVYEDGE_TEST_MODE=0 make -j ${MAKE_JOBS} test; then
    exit 3
fi

make_targets="examples"
case "${DOC_BUILD_MODE:-test}" in
  build)
    if ! HEAVYEDGE_TEST_MODE=0 make -j ${MAKE_JOBS} ${make_targets}; then
      exit 4
    fi
    ;;
  test)
    if ! HEAVYEDGE_TEST_MODE=1 make -j ${MAKE_JOBS} ${make_targets}; then
      exit 4
    fi
    ;;
  *)
    echo "::error::Unsupported doc build mode: ${DOC_BUILD_MODE}" >&2
    exit 4
    ;;
esac
