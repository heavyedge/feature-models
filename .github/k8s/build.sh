#!/bin/sh

set -eu

if ! ./setup.sh; then
  exit 1
fi

case "${BUILD_MODE:-test}" in
  build)
    if ! HEAVYEDGE_TEST_MODE=0 make -j ${MAKE_JOBS} models-${MAJOR_VERSION}; then
      exit 2
    fi
    if ! HEAVYEDGE_TEST_MODE=0 make -j ${MAKE_JOBS} test-${MAJOR_VERSION}; then
      exit 3
    fi
    if ! HEAVYEDGE_TEST_MODE=0 make -j ${MAKE_JOBS} examples-${MAJOR_VERSION}; then
      exit 4
    fi
    ;;
  test)
    if ! HEAVYEDGE_TEST_MODE=1 make -j ${MAKE_JOBS} models; then
      exit 2
    fi
    if ! HEAVYEDGE_TEST_MODE=1 make -j ${MAKE_JOBS} tests; then
      exit 3
    fi
    if ! HEAVYEDGE_TEST_MODE=1 make -j ${MAKE_JOBS} examples; then
      exit 4
    fi
    ;;
  *)
    echo "::error::Unsupported build mode: ${BUILD_MODE}" >&2
    exit 2
    ;;
esac
