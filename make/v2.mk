.PHONY: models-v2 examples-v2 test-v2

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

MODELS_v2 := \
models/v2/feature_models/prior_mean.pt \
models/v2/feature_models/gpr.pt

SCRIPTS_v2 := \
models/v2/feature_models/batch.py \
models/v2/feature_models/scale.py \
models/v2/feature_models/prior.py \
models/v2/feature_models/likelihoods.py \
models/v2/feature_models/gpr.py \
models/v2/feature_models/load.py \
models/v2/feature_models/predict-prior_mean.py

models-v2: $(MODELS_v2) $(SCRIPTS_v2)

examples-v2:

test-v2: $(MODELS_v2) $(SCRIPTS_v2)
	pytest tests/v2 --models-path $(CURDIR)/models/v2

# Data

_temp/v2/dimless.csv: $(wildcard _data/v1/dimless/all_profiles/dataset*.csv)
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

_temp/v2/shape_features.csv: $(wildcard _data/v1/shape_features/all_profiles/dataset*.csv)
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

_temp/v2/X.csv: scripts/v0/data/write-X.py _temp/v2/dimless.csv
	python3 $^ -o $@

_temp/v2/y.csv: scripts/v0/data/write-y.py _temp/v2/X.csv _temp/v2/shape_features.csv
	python3 $^ --index-col 0 1 2 -o $@

_temp/v2/Xsplit.csv: scripts/v0/data/split-X.py _temp/v2/X.csv
	python3 $^ --split-ratio 0.8 0.1 0.1 --num-folds 1 --random-state=42 -o $@

_temp/v2/ysplit.csv: scripts/v0/data/write-y.py _temp/v2/Xsplit.csv _temp/v2/shape_features.csv
	python3 $^ --index-col 0 1 2 3 4 -o $@

define SPLIT_v2
_temp/v2/X$(1).csv: _temp/v2/Xsplit.csv
	python3 -c "import pandas as pd; df = pd.read_csv('$$<'); mask = df['split'] == '$(1)'; df.loc[mask, ['fold', 'gap_to_thickness_ratio', 'capillary_number', 'cosine_of_contact_angle']].to_csv('$$@', index=False)"
_temp/v2/y$(1).csv: _temp/v2/ysplit.csv
	python3 -c "import pandas as pd; df = pd.read_csv('$$<'); mask = df['split'] == '$(1)'; df.loc[mask, ['fold', 'H', 'phi_1', 'phi_3']].to_csv('$$@', index=False)"
endef
$(foreach split,train val test,$(eval $(call SPLIT_v2,$(split))))

_temp/v2/Xpred_1D.csv: scripts/v0/data/write-Xpred.py _temp/v2/X.csv
	python3 $^ --target gap_to_thickness_ratio --ngrid $(N_GRID_1) -o $@

# Models

models/v2/feature_models/%.py: scripts/v2/model/%.py
	mkdir -p $(@D)
	cp $< $@

models/v2/feature_models/batch.py: scripts/v0/model/batch.py
	mkdir -p $(@D)
	cp $< $@

## Prior mean

_temp/v2/prior_mean.pt: scripts/v2/train/prior_mean.py _temp/v2/Xtrain.csv _temp/v2/ytrain.csv \
$(SCRIPTS_v2)
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,3,$^) --index-col 0 --batch-col 0 --model PriorMean --num-epochs $(N_EPOCHS) -o $@

models/v2/feature_models/prior_mean.pt: scripts/v2/train/prior_mean.py _temp/v2/X.csv _temp/v2/y.csv \
$(SCRIPTS_v2)
	mkdir -p $(@D)
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,3,$^) --index-col 0 1 2 --model PriorMean --num-epochs $(N_EPOCHS) -o $@

## GPR

_temp/v2/gpr_independent.pt: scripts/v2/train/gpr.py _temp/v2/Xtrain.csv _temp/v2/ytrain.csv _temp/v2/Xval.csv _temp/v2/yval.csv _temp/v2/prior_mean.pt \
$(SCRIPTS_v2)
	mkdir -p benchmarks
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,6,$^) --index-col 0 --batch-col 0 --model GPR_Independent --num-epochs $(N_EPOCHS) --n-trials=$(N_TRIALS) --storage=$(OPTUNA_DB) --study-name=v2/$*_independent.gpr -o $@

# Prediction

_temp/v2/prior_mean.Xpred_1D.csv: _temp/v2/Xpred_1D.csv $(SCRIPTS_v2) models/v2/feature_models/prior_mean.pt
	$(GPU_PYTHON) -m models.v2.feature_models.predict-prior_mean $< --index-col 0 1 2 -o $@

# Examples

examples/v2/Models.ipynb: \
_temp/v2/X.csv _temp/v2/y.csv _temp/v2/Xpred_1D.csv \
_temp/v2/prior_mean.Xpred_1D.csv \
.FORCE
	$(GPU_JUPYTER) nbconvert --to notebook --execute --inplace $@
