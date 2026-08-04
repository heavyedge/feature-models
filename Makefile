.ONESHELL:
.SECONDEXPANSION:
.SECONDARY:
.PHONY: all models examples tests clean .FORCE
# Dummy target to ensure that prerequisite files are built.
.FORCE:

all: models examples

models: models-v0

examples: examples-v0

tests: test-v0

clean:
	shopt -s globstar nullglob
	rm -rf _temp models/**/*.pt models/**/*.py

include make/v0.mk
