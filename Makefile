.ONESHELL:
.SECONDEXPANSION:
.SECONDARY:
.PHONY: all models examples tests clean .FORCE
# Dummy target to ensure that prerequisite files are built.
.FORCE:

PYTHON ?= python3
GPU_RUN ?= $(PYTHON) scripts/gpu-run.py --
GPU_PYTHON ?= $(GPU_RUN) $(PYTHON)
GPU_JUPYTER ?= $(GPU_RUN) jupyter

OPTUNA_DB ?= sqlite:///benchmarks/optuna.db

all: models examples

models: models-v0 models-v1 models-v2

examples: examples-v0 examples-v1 examples-v2

# Set PREBUILT_MODELS=1 to test existing model artifacts without rebuilding them.
tests: test-v0 test-v1 test-v2

clean:
	shopt -s globstar nullglob
	rm -rf _temp benchmarks models/**/*.pt models/**/*.py

include make/v0.mk
include make/v1.mk
include make/v2.mk
