.PHONY: models-v2 examples-v2 test-v2

N_FOLDS_v2 := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),2,5)
N_EPOCHS_v2 := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),1,15000)
N_TRIALS_v2 := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),1,100)

MODELS_v2 := \
models/v2/feature_models/prior_mean.pt \
models/v2/feature_models/gpr.pt \
models/v2/feature_models/gpqr.pt

SCRIPTS_v2 := \
models/v2/feature_models/batch.py \
models/v2/feature_models/scale.py \
models/v2/feature_models/prior.py \
models/v2/feature_models/gpr.py \
models/v2/feature_models/gpqr.py \
models/v2/feature_models/likelihoods.py \
models/v2/feature_models/load.py \
models/v2/feature_models/predict-prior_mean.py \
models/v2/feature_models/predict-gpr.py \
models/v2/feature_models/predict-gpqr.py

models-v2: $(MODELS_v2) $(SCRIPTS_v2)

examples-v2: $(wildcard examples/v2/*)

test-v2: $(if $(filter 1,$(PREBUILT_MODELS)),,$(MODELS_v2) $(SCRIPTS_v2))
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

# Models

models/v2/feature_models/%.py: scripts/v2/model/%.py
	mkdir -p $(@D)
	cp $< $@

models/v2/feature_models/batch.py: scripts/v0/model/batch.py
	mkdir -p $(@D)
	cp $< $@

## Prior mean

_temp/v2/prior_mean.pt: scripts/v2/train/prior_mean.py _temp/v1/Xtrain.csv _temp/v1/ytrain.csv \
$(SCRIPTS_v2)
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,3,$^) --index-col 0 --batch-col 0 --model PriorMean --num-epochs $(N_EPOCHS_v2) -o $@

models/v2/feature_models/prior_mean.pt: scripts/v2/train/prior_mean.py _temp/v1/X.csv _temp/v1/y.csv \
$(SCRIPTS_v2)
	mkdir -p $(@D)
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,3,$^) --index-col 0 1 2 --model PriorMean --num-epochs $(N_EPOCHS_v2) -o $@

## GPR

_temp/v2/gpr_independent.pt: scripts/v2/train/gpr.py _temp/v1/Xtrain.csv _temp/v1/ytrain.csv _temp/v1/Xval.csv _temp/v1/yval.csv _temp/v2/prior_mean.pt \
$(SCRIPTS_v2)
	mkdir -p benchmarks
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,6,$^) --index-col 0 --batch-col 0 --model GPR_Independent --num-epochs $(N_EPOCHS_v2) --optimize-hyperparameters noise_prior_loc noise_prior_scale lengthscale_prior_loc lengthscale_prior_scale --n-trials=$(N_TRIALS_v2) --storage=$(OPTUNA_DB) --study-name=v2/gpr.independent -o $@

_temp/v2/gpr_lmc.pt: scripts/v2/train/gpr.py _temp/v1/Xtrain.csv _temp/v1/ytrain.csv _temp/v1/Xval.csv _temp/v1/yval.csv _temp/v2/prior_mean.pt \
$(SCRIPTS_v2)
	mkdir -p benchmarks
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,6,$^) --index-col 0 --batch-col 0 --model GPR_LMC --num-epochs $(N_EPOCHS_v2) --optimize-hyperparameters noise_prior_loc noise_prior_scale lengthscale_prior_loc lengthscale_prior_scale num_latents --n-trials=$(N_TRIALS_v2) --storage=$(OPTUNA_DB) --study-name=v2/gpr.lmc -o $@

models/v2/feature_models/gpr.pt: scripts/v2/train/gpr.py _temp/v1/X.csv _temp/v1/y.csv models/v2/feature_models/prior_mean.pt \
_temp/v2/gpr_independent.pt $(SCRIPTS_v2)
	mkdir -p $(@D)
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,4,$^) --index-col 0 1 2 --model GPR_Independent --storage=$(OPTUNA_DB) --study-name=v2/gpr.independent -o $@

## GPQR

_temp/v2/gpqr_independent.pt: scripts/v2/train/gpqr.py _temp/v1/Xtrain.csv _temp/v1/ytrain.csv _temp/v1/Xval.csv _temp/v1/yval.csv _temp/v2/prior_mean.pt \
$(SCRIPTS_v2)
	mkdir -p benchmarks
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,6,$^) --index-col 0 --batch-col 0 --model GPQR_Independent --quantiles $(QUANTILES) --num-likelihood-samples $(N_LIKELIHOOD_SAMPLES) --num-epochs $(N_EPOCHS_v2) --early-stopping-patience-ratio 0.05 --optimize-hyperparameters noise_prior_loc noise_prior_scale lengthscale_prior_loc lengthscale_prior_scale --n-trials=$(N_TRIALS_v2) --storage=$(OPTUNA_DB) --study-name=v2/gpqr_independent -o $@

_temp/v2/gpqr_lmc.pt: scripts/v2/train/gpqr.py _temp/v1/Xtrain.csv _temp/v1/ytrain.csv _temp/v1/Xval.csv _temp/v1/yval.csv _temp/v2/prior_mean.pt \
$(SCRIPTS_v2)
	mkdir -p benchmarks
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,6,$^) --index-col 0 --batch-col 0 --model GPQR_LMC --quantiles $(QUANTILES) --num-likelihood-samples $(N_LIKELIHOOD_SAMPLES) --num-epochs $(N_EPOCHS_v2) --early-stopping-patience-ratio 0.05 --optimize-hyperparameters noise_prior_loc noise_prior_scale lengthscale_prior_loc lengthscale_prior_scale num_latents --n-trials=$(N_TRIALS_v2) --storage=$(OPTUNA_DB) --study-name=v2/gpqr_lmc -o $@

_temp/v2/gpqr_cglmc.pt: scripts/v2/train/gpqr.py _temp/v1/Xtrain.csv _temp/v1/ytrain.csv _temp/v1/Xval.csv _temp/v1/yval.csv _temp/v2/prior_mean.pt \
$(SCRIPTS_v2)
	mkdir -p benchmarks
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,6,$^) --index-col 0 --batch-col 0 --model GPQR_CenterGapLMC --quantiles $(QUANTILES) --num-likelihood-samples $(N_LIKELIHOOD_SAMPLES) --num-epochs $(N_EPOCHS_v2) --early-stopping-patience-ratio 0.05 --optimize-hyperparameters noise_prior_loc noise_prior_scale lengthscale_prior_loc lengthscale_prior_scale num_latents num_central_latents --n-trials=$(N_TRIALS_v2) --storage=$(OPTUNA_DB) --study-name=v2/gpqr_cglmc -o $@

models/v2/feature_models/gpqr.pt: scripts/v2/train/gpqr.py _temp/v1/X.csv _temp/v1/y.csv models/v2/feature_models/prior_mean.pt \
_temp/v2/gpqr_independent.pt $(SCRIPTS_v2)
	mkdir -p $(@D)
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,4,$^) --index-col 0 1 2 --model GPQR_Independent --quantiles $(QUANTILES) --num-likelihood-samples $(N_LIKELIHOOD_SAMPLES) --storage=$(OPTUNA_DB) --study-name=v2/gpqr_independent -o $@

# Prediction

_temp/v2/prior_mean.Xpred_1D.csv: _temp/v1/Xpred_1D.csv $(SCRIPTS_v2) models/v2/feature_models/prior_mean.pt
	$(GPU_PYTHON) -m models.v2.feature_models.predict-prior_mean $< --index-col 0 1 2 -o $@

_temp/v2/gpr.Xpred_1D.csv: _temp/v1/Xpred_1D.csv $(SCRIPTS_v2) models/v2/feature_models/prior_mean.pt models/v2/feature_models/gpr.pt
	$(GPU_PYTHON) -m models.v2.feature_models.predict-gpr $< --index-col 0 1 2 -o $@

_temp/v2/gpr_%.Xtest.csv: _temp/v1/Xtest.csv _temp/v2/prior_mean.pt _temp/v2/gpr_%.pt $(SCRIPTS_v2)
	$(GPU_PYTHON) -m models.v2.feature_models.predict-gpr $(wordlist 1,3,$^) --index-col 0 --batch-col 0 -o $@

_temp/v2/gpqr.Xpred_1D.csv: _temp/v1/Xpred_1D.csv $(SCRIPTS_v2) models/v2/feature_models/prior_mean.pt models/v2/feature_models/gpqr.pt
	$(GPU_PYTHON) -m models.v2.feature_models.predict-gpqr $< --index-col 0 1 2 -o $@

_temp/v2/gpqr_%.Xtest.csv: _temp/v1/Xtest.csv _temp/v2/prior_mean.pt _temp/v2/gpqr_%.pt $(SCRIPTS_v2)
	$(GPU_PYTHON) -m models.v2.feature_models.predict-gpqr $(wordlist 1,3,$^) --index-col 0 --batch-col 0 -o $@

# Model selection

benchmarks/v2/rmse.gpr_%.csv: scripts/v2/model_selection/rmse.py _temp/v2/gpr_%.Xtest.csv _temp/v1/ytest.csv
	mkdir -p $(@D)
	python3 $^ --index-col 0 -o $@

benchmarks/v2/nlpd.gpr_%.csv: scripts/v2/model_selection/nlpd.py _temp/v2/gpr_%.Xtest.csv _temp/v1/ytest.csv
	mkdir -p $(@D)
	python3 $^ --index-col 0 -o $@

benchmarks/v2/pinball_loss.gpr_%.csv: scripts/v2/model_selection/pinball_loss.py _temp/v2/gpr_%.Xtest.csv _temp/v1/ytest.csv
	mkdir -p $(@D)
	python3 $^ --type GPR --quantile-levels $(QUANTILES) -o $@

benchmarks/v2/pinball_loss.gpqr_%.csv: scripts/v2/model_selection/pinball_loss.py _temp/v2/gpqr_%.Xtest.csv _temp/v1/ytest.csv
	mkdir -p $(@D)
	python3 $^ --type GPQR --quantile-levels $(QUANTILES) -o $@

# Examples

examples/v2/Evaluation.ipynb: \
benchmarks/v2/rmse.gpr_independent.csv benchmarks/v2/rmse.gpr_lmc.csv \
benchmarks/v2/nlpd.gpr_independent.csv benchmarks/v2/nlpd.gpr_lmc.csv \
benchmarks/v2/pinball_loss.gpr_independent.csv benchmarks/v2/pinball_loss.gpr_lmc.csv \
benchmarks/v2/pinball_loss.gpqr_independent.csv benchmarks/v2/pinball_loss.gpqr_lmc.csv benchmarks/v2/pinball_loss.gpqr_cglmc.csv \
.FORCE
	$(GPU_JUPYTER) nbconvert --to notebook --execute --inplace $@

examples/v2/Models.ipynb: \
_temp/v1/X.csv _temp/v1/y.csv _temp/v1/Xpred_1D.csv \
_temp/v2/prior_mean.Xpred_1D.csv \
_temp/v2/gpr.Xpred_1D.csv \
_temp/v2/gpqr.Xpred_1D.csv \
.FORCE
	$(GPU_JUPYTER) nbconvert --to notebook --execute --inplace $@
