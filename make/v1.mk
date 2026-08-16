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
models/v1/feature_models/phi.gpr.pt \
models/v1/feature_models/H.gpqr.pt \
models/v1/feature_models/phi.gpqr.pt

SCRIPTS_v1 := \
models/v1/feature_models/prior.py \
models/v1/feature_models/scale.py \
models/v1/feature_models/gpr.py \
models/v1/feature_models/gpqr.py \
models/v1/feature_models/likelihoods.py \
models/v1/feature_models/load.py \
models/v1/feature_models/batch.py \
models/v1/feature_models/predict-prior_mean.py \
models/v1/feature_models/predict-gpr.py \
models/v1/feature_models/predict-gpqr.py

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

_temp/v1/Xpred_1D.csv: scripts/v0/data/write-Xpred.py _temp/v1/X.csv
	python3 $^ --target gap_to_thickness_ratio --ngrid $(N_GRID_1) -o $@

# Models

models/v1/feature_models/%.py: scripts/v0/model/%.py
	mkdir -p $(@D)
	cp $< $@

models/v1/feature_models/gpr.py: scripts/v1/model/gpr.py
	mkdir -p $(@D)
	cp $< $@

models/v1/feature_models/load.py: scripts/v1/model/load.py
	mkdir -p $(@D)
	cp $< $@

## Prior mean

_temp/v1/%.prior_mean.pt: scripts/v0/train/prior_mean.py _temp/v1/Xtrain.csv _temp/v1/ytrain.csv \
$(SCRIPTS_v0)
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,3,$^) --index-col 0 --batch-col 0 --target $* --model PriorMean_$* --num-epochs $(N_EPOCHS) -o $@

models/v1/feature_models/%.prior_mean.pt: scripts/v0/train/prior_mean.py _temp/v1/X.csv _temp/v1/y.csv \
$(SCRIPTS_v0)
	mkdir -p $(@D)
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,3,$^) --index-col 0 1 2 --target $* --model PriorMean_$* --num-epochs $(N_EPOCHS) -o $@

## GPR

_temp/v1/%.gpr.pt: scripts/v1/train/gpr.py _temp/v1/Xtrain.csv _temp/v1/ytrain.csv _temp/v1/Xval.csv _temp/v1/yval.csv _temp/v1/%.prior_mean.pt \
$(SCRIPTS_v1)
	mkdir -p benchmarks
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,6,$^) --index-col 0 --batch-col 0 --target $* --model GPR_$* --num-epochs $(N_EPOCHS) --n-trials=$(N_TRIALS) --storage=$(OPTUNA_DB) --study-name=v1/$*.gpr -o $@

models/v1/feature_models/%.gpr.pt: scripts/v1/train/gpr.py _temp/v1/X.csv _temp/v1/y.csv models/v1/feature_models/%.prior_mean.pt \
_temp/v1/%.gpr.pt $(SCRIPTS_v0)
	mkdir -p $(@D)
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,4,$^) --index-col 0 1 2 --target $* --model GPR_$* --storage=$(OPTUNA_DB) --study-name=v1/$*.gpr -o $@

## GPQR

_temp/v1/%.cg_gpqr_independent.pt: scripts/v0/train/gpqr.py _temp/v1/Xtrain.csv _temp/v1/ytrain.csv _temp/v1/Xval.csv _temp/v1/yval.csv _temp/v1/%.prior_mean.pt \
$(SCRIPTS_v0)
	mkdir -p benchmarks
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,6,$^) --index-col 0 --batch-col 0 --target $* --model CenterGapMTGPQR_Independent_$* --quantiles $(QUANTILES) --num-likelihood-samples $(N_LIKELIHOOD_SAMPLES) --num-epochs $(N_EPOCHS) --n-trials=$(N_TRIALS) --storage=$(OPTUNA_DB) --study-name=v1/$*.cg_gpqr_independent -o $@

_temp/v1/%.cg_gpqr_lmc.pt: scripts/v0/train/gpqr.py _temp/v1/Xtrain.csv _temp/v1/ytrain.csv _temp/v1/Xval.csv _temp/v1/yval.csv _temp/v1/%.prior_mean.pt \
$(SCRIPTS_v0)
	mkdir -p benchmarks
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,6,$^) --index-col 0 --batch-col 0 --target $* --model CenterGapMTGPQR_LMC_$* --quantiles $(QUANTILES) --num-likelihood-samples $(N_LIKELIHOOD_SAMPLES) --num-epochs $(N_EPOCHS) --n-trials=$(N_TRIALS) --storage=$(OPTUNA_DB) --study-name=v1/$*.cg_gpqr_lmc -o $@

_temp/v1/%.cg_gpqr_cglmc.pt: scripts/v0/train/gpqr.py _temp/v1/Xtrain.csv _temp/v1/ytrain.csv _temp/v1/Xval.csv _temp/v1/yval.csv _temp/v1/%.prior_mean.pt \
$(SCRIPTS_v0)
	mkdir -p benchmarks
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,6,$^) --index-col 0 --batch-col 0 --target $* --model CenterGapMTGPQR_CenterGapLMC_$* --quantiles $(QUANTILES) --num-likelihood-samples $(N_LIKELIHOOD_SAMPLES) --num-epochs $(N_EPOCHS) --n-trials=$(N_TRIALS) --storage=$(OPTUNA_DB) --study-name=v1/$*.cg_gpqr_cglmc -o $@

models/v1/feature_models/%.gpqr.pt: scripts/v0/train/gpqr.py _temp/v1/X.csv _temp/v1/y.csv models/v1/feature_models/%.prior_mean.pt \
_temp/v1/%.cg_gpqr_cglmc.pt $(SCRIPTS_v0)
	mkdir -p $(@D)
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,4,$^) --index-col 0 1 2 --target $* --model CenterGapMTGPQR_CenterGapLMC_$* --quantiles $(QUANTILES) --num-likelihood-samples $(N_LIKELIHOOD_SAMPLES) --storage=$(OPTUNA_DB) --study-name=v1/$*.cg_gpqr_cglmc -o $@

# Prediction

_temp/v1/%.prior_mean.Xpred_1D.csv: _temp/v1/Xpred_1D.csv $(SCRIPTS_v1) models/v1/feature_models/%.prior_mean.pt
	$(GPU_PYTHON) -m models.v1.feature_models.predict-prior_mean $< --index-col 0 1 2 --target $* -o $@

_temp/v1/%.gpr.Xpred_1D.csv: _temp/v1/Xpred_1D.csv $(SCRIPTS_v1) models/v1/feature_models/%.prior_mean.pt models/v1/feature_models/%.gpr.pt
	$(GPU_PYTHON) -m models.v1.feature_models.predict-gpr $< --index-col 0 1 2 --target $* -o $@

_temp/v1/%.gpr.Xtest.csv: _temp/v1/Xtest.csv _temp/v1/%.prior_mean.pt _temp/v1/%.gpr.pt $(SCRIPTS_v1)
	$(GPU_PYTHON) -m models.v1.feature_models.predict-gpr $(wordlist 1,3,$^) --index-col 0 --batch-col 0 --target $* -o $@

_temp/v1/%.gpqr.Xpred_1D.csv: _temp/v1/Xpred_1D.csv $(SCRIPTS_v1) models/v1/feature_models/%.prior_mean.pt models/v1/feature_models/%.gpqr.pt
	$(GPU_PYTHON) -m models.v1.feature_models.predict-gpqr $< --index-col 0 1 2 --target $* -o $@

# Model selection

benchmarks/v1/pinball_loss.%.gpr.csv: scripts/v0/model_selection/pinball_loss.py _temp/v1/%.gpr.Xtest.csv _temp/v1/ytest.csv
	mkdir -p $(@D)
	python3 $^ --type GPR --quantile-levels $(QUANTILES) -o $@

# Examples

examples/v1/CV.ipynb: \
benchmarks/v1/pinball_loss.H.gpr.csv benchmarks/v1/pinball_loss.phi.gpr.csv \
.FORCE
	$(GPU_JUPYTER) nbconvert --to notebook --execute --inplace $@

examples/v1/Models.ipynb: \
_temp/v1/X.csv _temp/v1/y.csv _temp/v1/Xpred_1D.csv \
_temp/v1/H.prior_mean.Xpred_1D.csv _temp/v1/phi.prior_mean.Xpred_1D.csv \
_temp/v1/H.gpr.Xpred_1D.csv _temp/v1/phi.gpr.Xpred_1D.csv \
_temp/v1/H.gpqr.Xpred_1D.csv _temp/v1/phi.gpqr.Xpred_1D.csv \
.FORCE
	$(GPU_JUPYTER) nbconvert --to notebook --execute --inplace $@
