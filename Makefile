.ONESHELL:
.SECONDEXPANSION:
.SECONDARY:
.PHONY: all models examples tests clean .FORCE
# Dummy target to ensure that prerequisite files are built.
.FORCE:

all: models examples

models: models-v0 models-v1

examples: examples-v0 examples-v1

tests: test-v0

clean:
	shopt -s globstar nullglob
	rm -rf _temp models/**/*.pt models/**/*.py

include make/v0.mk
include make/v1.mk
