.PHONY: models-v1 examples-v1 test-v1

QUANTILES := 0.05 0.25 0.5 0.75 0.95
NUM_LOWER_QUANTILES := 2
NUM_LATENTS := 3

N_EPOCHS := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),1,10000)
N_FOLDS := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),2,10)
N_GRID_1 := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),2,200)
N_GRID_2 := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),2,10)
N_TRIALS := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),1,100)

H_THRESHOLD := 1.1
PHI_THRESHOLD := 1.0

OPTUNA_DB ?= sqlite:///benchmarks/v1/optuna.db

MODEL_FILES_v1 := \
models/v1/feature_models/H.prior_mean.pt \
models/v1/feature_models/phi.prior_mean.pt \
models/v1/feature_models/H.gpr.pt \
models/v1/feature_models/phi.gpr.pt \
models/v1/feature_models/prior.py \
models/v1/feature_models/scale.py \
models/v1/feature_models/gpr.py \
models/v1/feature_models/load.py \
models/v1/feature_models/predict-prior_mean.py \
models/v1/feature_models/predict-mean.py

models-v1: $(MODEL_FILES_v1)

examples-v1: $(wildcard examples/v1/*)

test-v1: $(MODEL_FILES_v1)
	$(GPU_PYTHON) -c "from models.v1.feature_models.load import load_PriorMean_H; load_PriorMean_H()"
	$(GPU_PYTHON) -c "from models.v1.feature_models.load import load_PriorMean_phi; load_PriorMean_phi()"
	$(GPU_PYTHON) -c "from models.v1.feature_models.load import load_GPR_H; load_GPR_H()"
	$(GPU_PYTHON) -c "from models.v1.feature_models.load import load_GPR_phi; load_GPR_phi()"

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

_temp/v1/X.csv: scripts/v1/data/write-X.py _temp/v1/dimless.csv
	mkdir -p $(@D)
	python3 $^ --split-ratio 0.8 0.1 0.1 --random-state=42 -o $@

_temp/v1/y.csv: scripts/v1/data/write-y.py _temp/v1/X.csv _temp/v1/shape_features.csv
	mkdir -p $(@D)
	python3 $^ -o $@

define SPLIT_v1
_temp/v1/X$(1).csv: _temp/v1/X.csv
	python3 -c "import pandas as pd; df = pd.read_csv('$$<'); mask = df['split'] == '$(1)'; df.loc[mask, ['gap_to_thickness_ratio', 'capillary_number', 'cosine_of_contact_angle']].to_csv('$$@', index=False)"
_temp/v1/y$(1).csv: _temp/v1/y.csv
	python3 -c "import pandas as pd; df = pd.read_csv('$$<'); mask = df['split'] == '$(1)'; df.loc[mask, ['H', 'phi']].to_csv('$$@', index=False)"
endef
$(foreach split,train val test,$(eval $(call SPLIT_v1,$(split))))

# Models

## Prior mean

models/v1/feature_models/%.prior_mean.pt: scripts/v1/train/prior_mean.py _temp/v1/Xtrain.csv _temp/v1/ytrain.csv _temp/v1/Xval.csv _temp/v1/yval.csv
	mkdir -p $(@D)
	PYTHONPATH=scripts $(GPU_PYTHON) $^ --target $* --model PriorMean_$* --num-epochs $(N_EPOCHS) -o $@

models/v1/feature_models/%.gpr.pt: scripts/v1/train/gpr.py _temp/v1/Xtrain.csv _temp/v1/ytrain.csv _temp/v1/Xval.csv _temp/v1/yval.csv models/v1/feature_models/%.prior_mean.pt
	mkdir -p $(@D) benchmarks/v1
	PYTHONPATH=scripts $(GPU_PYTHON) $^ --target $* --model GPR_$* --num-epochs $(N_EPOCHS) --n-trials=$(N_TRIALS) --storage=$(OPTUNA_DB) -o $@

models/v1/feature_models/%.py: scripts/v0/model/%.py
	mkdir -p $(@D)
	cp $< $@

models/v1/feature_models/gpr.py: scripts/v1/model/gpr.py
	mkdir -p $(@D)
	cp $< $@

models/v1/feature_models/load.py: scripts/v1/model/load.py
	mkdir -p $(@D)
	cp $< $@

# Prediction

_temp/v1/%.prior_mean.Xpred_1D.csv: models/v1/feature_models/predict-prior_mean.py _temp/v0/Xpred_1D.csv $(MODEL_FILES_v1)
	$(GPU_PYTHON) $(wordlist 1,2,$^) --target $* -o $@

_temp/v1/%.mean.Xpred_1D.csv: models/v1/feature_models/predict-mean.py _temp/v0/Xpred_1D.csv $(MODEL_FILES_v1)
	$(GPU_PYTHON) $(wordlist 1,2,$^) --target $* -o $@

# Examples

examples/v1/Models.ipynb: _temp/v1/X.csv _temp/v1/y.csv _temp/v0/Xpred_1D.csv _temp/v1/H.prior_mean.Xpred_1D.csv _temp/v1/phi.prior_mean.Xpred_1D.csv _temp/v1/H.mean.Xpred_1D.csv _temp/v1/phi.mean.Xpred_1D.csv .FORCE
	$(GPU_JUPYTER) nbconvert --to notebook --execute --inplace $@
