export HEAVYEDGE_GPU_DEVICES="$(python3 scripts/cuda-preflight.py --print-devices)"
export MAKE_JOBS="$(python3 scripts/cuda-preflight.py --print-count)"
python3 scripts/cuda-preflight.py

if [ "${GITHUB_EVENT_NAME}" = "release" ]; then
    HEAVYEDGE_TEST_MODE=0 make -j "$MAKE_JOBS" "models-${MAJOR_VERSION}"
    HEAVYEDGE_TEST_MODE=0 make -j "$MAKE_JOBS" "examples-${MAJOR_VERSION}"
else
    HEAVYEDGE_TEST_MODE=1 make -j "$MAKE_JOBS" models
    HEAVYEDGE_TEST_MODE=1 make -j "$MAKE_JOBS" tests
    HEAVYEDGE_TEST_MODE=1 make -j "$MAKE_JOBS" examples
fi