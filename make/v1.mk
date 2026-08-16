.PHONY: models-v1 examples-v1 test-v1

QUANTILES := 0.05 0.25 0.5 0.75 0.95
NUM_LOWER_QUANTILES := 2
NUM_LATENTS := 3

N_LIKELIHOOD_SAMPLES := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),4,64)
N_EPOCHS := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),1,10000)
N_GRID_1 := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),2,200)
N_GRID_2 := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),2,10)
N_TRIALS := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),1,100)

H_THRESHOLD := 1.1
PHI_THRESHOLD := 1.0

MODELS_v1 := \
models/v1/feature_models/H.prior_mean.pt \
models/v1/feature_models/phi.prior_mean.pt \
models/v1/feature_models/H.gpr.pt \
models/v1/feature_models/phi.gpr.pt

SCRIPTS_v1 := \
models/v1/feature_models/prior.py \
models/v1/feature_models/scale.py \
models/v1/feature_models/gpr.py \
models/v1/feature_models/load.py \
models/v1/feature_models/predict-prior_mean.py \
models/v1/feature_models/predict-gpr.py

models-v1: $(MODELS_v1) $(SCRIPTS_v1)

examples-v1: $(wildcard examples/v1/*)

test-v1:

# Data

_temp/v1/dimless.csv: $(wildcard _data/v1/dimless/all_profiles/dataset*.csv)
	mkdir -p $(@D)
	python3 -c '
	from pathlib import Path
	import pandas as pd

	pvs = sorted([Path(f) for f in "$^".split(" ")])
	df = pd.concat(
		[pd.read_csv(f, dtype=str).assign(name=lambda df: f"{f.stem}/" + df["name"]) for f in pvs],
		ignore_index=True,
	)
	df.to_csv("$@", index=False)
	'

_temp/v1/shape_features.csv: $(wildcard _data/v1/shape_features/profiles/dataset*.csv)
	mkdir -p $(@D)
	python3 -c '
	from pathlib import Path
	import pandas as pd

	pvs = sorted([Path(f) for f in "$^".split(" ")])
	df = pd.concat(
		[pd.read_csv(f, dtype=str).assign(name=lambda df: f"{f.stem}/" + df["name"]) for f in pvs],
		ignore_index=True,
	)
	df.to_csv("$@", index=False)
	'

_temp/v1/X.csv: scripts/v0/data/write-X.py _temp/v1/dimless.csv
	python3 $^ -o $@

_temp/v1/y.csv: scripts/v0/data/write-y.py _temp/v1/X.csv _temp/v1/shape_features.csv
	python3 $^ --index-col 0 1 2 -o $@

_temp/v1/Xsplit.csv: scripts/v0/data/split-X.py _temp/v1/X.csv
	python3 $^ --split-ratio 0.8 0.1 0.1 --num-folds 1 --random-state=42 -o $@

_temp/v1/ysplit.csv: scripts/v0/data/write-y.py _temp/v1/Xsplit.csv _temp/v1/shape_features.csv
	python3 $^ --index-col 0 1 2 3 4 -o $@

define SPLIT_v1
_temp/v1/X$(1).csv: _temp/v1/Xsplit.csv
	python3 -c "import pandas as pd; df = pd.read_csv('$$<'); mask = df['split'] == '$(1)'; df.loc[mask, ['fold', 'gap_to_thickness_ratio', 'capillary_number', 'cosine_of_contact_angle']].to_csv('$$@', index=False)"
_temp/v1/y$(1).csv: _temp/v1/ysplit.csv
	python3 -c "import pandas as pd; df = pd.read_csv('$$<'); mask = df['split'] == '$(1)'; df.loc[mask, ['fold', 'H', 'phi']].to_csv('$$@', index=False)"
endef
$(foreach split,train val test,$(eval $(call SPLIT_v1,$(split))))
