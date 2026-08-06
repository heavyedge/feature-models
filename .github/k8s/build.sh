#!/bin/sh

set -eu

if ! ./setup.sh; then
  exit 1
fi

if ! sh .github/k8s/setup-postgres.sh; then
  exit 1
fi

if ! HEAVYEDGE_GPU_DEVICES=$(python3 scripts/cuda-preflight.py --print-devices); then
  exit 2
fi
export HEAVYEDGE_GPU_DEVICES
if [ -z "${MAKE_JOBS:-}" ] || [ "${MAKE_JOBS}" = "auto" ]; then
  if ! MAKE_JOBS=$(python3 scripts/cuda-preflight.py --print-count); then
    exit 3
  fi
fi
export MAKE_JOBS
if ! python3 scripts/cuda-preflight.py; then
  exit 4
fi

case "${BUILD_MODE:-test}" in
  build)
    if ! HEAVYEDGE_TEST_MODE=0 make -j ${MAKE_JOBS} models-${MAJOR_VERSION}; then
      exit 5
    fi
    if ! HEAVYEDGE_TEST_MODE=0 make -j ${MAKE_JOBS} test-${MAJOR_VERSION}; then
      exit 6
    fi
    if ! HEAVYEDGE_TEST_MODE=0 make -j ${MAKE_JOBS} examples-${MAJOR_VERSION}; then
      exit 7
    fi
    ;;
  test)
    if ! HEAVYEDGE_TEST_MODE=1 make -j ${MAKE_JOBS} models; then
      exit 5
    fi
    if ! HEAVYEDGE_TEST_MODE=1 make -j ${MAKE_JOBS} tests; then
      exit 6
    fi
    if ! HEAVYEDGE_TEST_MODE=1 make -j ${MAKE_JOBS} examples; then
      exit 7
    fi
    ;;
  *)
    echo "::error::Unsupported build mode: ${BUILD_MODE}" >&2
    exit 5
    ;;
esac
