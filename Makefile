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

QUANTILES := 0.05 0.25 0.5 0.75 0.95
NUM_LOWER_QUANTILES := 2
NUM_LATENTS := 3

N_LIKELIHOOD_SAMPLES := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),4,64)
N_GRID_1 := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),2,200)
N_GRID_2 := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),2,10)

H_THRESHOLD := 1.1
PHI_THRESHOLD := 0.25

all: models examples

models: models-v0 models-v1

examples: examples-v0 examples-v1

# Set PREBUILT_MODELS=1 to test existing model artifacts without rebuilding them.
tests: test-v0 test-v1

clean:
	shopt -s globstar nullglob
	rm -rf _temp benchmarks models/**/*.pt models/**/*.py

include make/v0.mk
include make/v1.mk
