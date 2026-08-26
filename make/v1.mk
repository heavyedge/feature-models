.PHONY: models-v1 examples-v1 test-v1

N_EPOCHS_v1 := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),1,10000)
N_TRIALS_v1 := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),1,10)

MODELS_v1 := \
models/v1/feature_models/prior_mean.pt \
models/v1/feature_models/gpr.pt \
models/v1/feature_models/gpqr.pt

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

test-v1: $(if $(filter 1,$(PREBUILT_MODELS)),,$(MODELS_v1) $(SCRIPTS_v1))
	pytest tests/v1 --models-path $(CURDIR)/models/v1

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

_temp/v1/shape_features.csv: $(wildcard _data/v1/shape_features/all_profiles/dataset*.csv)
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
	python3 -c "import pandas as pd; df = pd.read_csv('$$<'); mask = df['split'] == '$(1)'; df.loc[mask, ['fold', 'H', 'phi_1', 'phi_3']].to_csv('$$@', index=False)"
endef
$(foreach split,train val test,$(eval $(call SPLIT_v1,$(split))))

_temp/v1/Xpred_1D.csv: scripts/v0/data/write-Xpred.py _temp/v1/X.csv
	python3 $^ --target gap_to_thickness_ratio --ngrid $(N_GRID_1) -o $@

_temp/v1/Xpred_2D.csv: scripts/v0/data/write-Xpred.py _temp/v1/X.csv
	python3 $^ --target gap_to_thickness_ratio capillary_number --ngrid $(N_GRID_1) -o $@

_temp/v1/delaunay.Xpred_2D.csv: scripts/v0/data/compute-Delaunay.py _temp/v1/X.csv _temp/v1/Xpred_2D.csv
	python3 $^ --grid gap_to_thickness_ratio capillary_number -o $@

# Models

models/v1/feature_models/%.py: scripts/v1/model/%.py
	mkdir -p $(@D)
	cp $< $@

## Prior mean

_temp/v1/prior_mean.pt: scripts/v1/train/prior_mean.py _temp/v1/Xtrain.csv _temp/v1/ytrain.csv \
$(SCRIPTS_v1)
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,3,$^) --index-col 0 --batch-col 0 --model PriorMean --num-epochs $(N_EPOCHS_v1) -o $@

models/v1/feature_models/prior_mean.pt: scripts/v1/train/prior_mean.py _temp/v1/X.csv _temp/v1/y.csv \
$(SCRIPTS_v1)
	mkdir -p $(@D)
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,3,$^) --index-col 0 1 2 --model PriorMean --num-epochs $(N_EPOCHS_v1) -o $@

## GPR

_temp/v1/gpr.pt: scripts/v1/train/gpr.py _temp/v1/Xtrain.csv _temp/v1/ytrain.csv _temp/v1/Xval.csv _temp/v1/yval.csv _temp/v1/prior_mean.pt \
$(SCRIPTS_v1)
	mkdir -p benchmarks
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,6,$^) --index-col 0 --batch-col 0 --model GPR --num-epochs $(N_EPOCHS_v1) --optimize-hyperparameters lengthscale_prior_loc lengthscale_prior_scale --n-trials=$(N_TRIALS_v1) --storage=$(OPTUNA_DB) --study-name=v1/gpr -o $@

models/v1/feature_models/gpr.pt: scripts/v1/train/gpr.py _temp/v1/X.csv _temp/v1/y.csv models/v1/feature_models/prior_mean.pt \
_temp/v1/gpr.pt $(SCRIPTS_v1)
	mkdir -p $(@D)
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,4,$^) --index-col 0 1 2 --model GPR --storage=$(OPTUNA_DB) --study-name=v1/gpr -o $@

## GPQR

_temp/v1/gpqr_independent.pt: scripts/v1/train/gpqr.py _temp/v1/Xtrain.csv _temp/v1/ytrain.csv _temp/v1/Xval.csv _temp/v1/yval.csv _temp/v1/prior_mean.pt \
$(SCRIPTS_v1)
	mkdir -p benchmarks
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,6,$^) --index-col 0 --batch-col 0 --model GPQR_Independent --quantiles $(QUANTILES) --num-likelihood-samples $(N_LIKELIHOOD_SAMPLES) --num-epochs $(N_EPOCHS_v1) --optimize-hyperparameters lengthscale_prior_loc lengthscale_prior_scale --n-trials=$(N_TRIALS_v1) --storage=$(OPTUNA_DB) --study-name=v1/gpqr_independent -o $@

_temp/v1/gpqr_lmc.pt: scripts/v1/train/gpqr.py _temp/v1/Xtrain.csv _temp/v1/ytrain.csv _temp/v1/Xval.csv _temp/v1/yval.csv _temp/v1/prior_mean.pt \
$(SCRIPTS_v1)
	mkdir -p benchmarks
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,6,$^) --index-col 0 --batch-col 0 --model GPQR_LMC --quantiles $(QUANTILES) --num-likelihood-samples $(N_LIKELIHOOD_SAMPLES) --num-epochs $(N_EPOCHS_v1) --optimize-hyperparameters lengthscale_prior_loc lengthscale_prior_scale --n-trials=$(N_TRIALS_v1) --storage=$(OPTUNA_DB) --study-name=v1/gpqr_lmc -o $@

_temp/v1/gpqr_cglmc.pt: scripts/v1/train/gpqr.py _temp/v1/Xtrain.csv _temp/v1/ytrain.csv _temp/v1/Xval.csv _temp/v1/yval.csv _temp/v1/prior_mean.pt \
$(SCRIPTS_v1)
	mkdir -p benchmarks
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,6,$^) --index-col 0 --batch-col 0 --model GPQR_CenterGapLMC --quantiles $(QUANTILES) --num-likelihood-samples $(N_LIKELIHOOD_SAMPLES) --num-epochs $(N_EPOCHS_v1) --optimize-hyperparameters lengthscale_prior_loc lengthscale_prior_scale --n-trials=$(N_TRIALS_v1) --storage=$(OPTUNA_DB) --study-name=v1/gpqr_cglmc -o $@

models/v1/feature_models/gpqr.pt: scripts/v1/train/gpqr.py _temp/v1/X.csv _temp/v1/y.csv models/v1/feature_models/prior_mean.pt \
_temp/v1/gpqr_cglmc.pt $(SCRIPTS_v1)
	mkdir -p $(@D)
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,4,$^) --index-col 0 1 2 --model GPQR_CenterGapLMC --quantiles $(QUANTILES) --num-likelihood-samples $(N_LIKELIHOOD_SAMPLES) --storage=$(OPTUNA_DB) --study-name=v1/gpqr_cglmc -o $@

# Prediction

benchmarks/v1/prior_mean.Xpred_1D.csv: _temp/v1/Xpred_1D.csv $(SCRIPTS_v1) models/v1/feature_models/prior_mean.pt
	mkdir -p $(@D)
	$(GPU_PYTHON) -m models.v1.feature_models.predict-prior_mean $< --index-col 0 1 2 -o $@

benchmarks/v1/gpr.Xpred_1D.csv: _temp/v1/Xpred_1D.csv $(SCRIPTS_v1) models/v1/feature_models/prior_mean.pt models/v1/feature_models/gpr.pt
	mkdir -p $(@D)
	$(GPU_PYTHON) -m models.v1.feature_models.predict-gpr $< --index-col 0 1 2 -o $@

benchmarks/v1/gpr.Xpred_2D.csv:
	mkdir -p $(@D)
	$(GPU_PYTHON) -m models.v1.feature_models.predict-gpr _temp/v1/Xpred_2D.csv --index-col 0 1 2 -o $@

benchmarks/v1/gpr.Xtest.csv: _temp/v1/Xtest.csv _temp/v1/prior_mean.pt _temp/v1/gpr.pt $(SCRIPTS_v1)
	mkdir -p $(@D)
	$(GPU_PYTHON) -m models.v1.feature_models.predict-gpr $(wordlist 1,3,$^) --index-col 0 --batch-col 0 -o $@

benchmarks/v1/gpqr.X.csv: _temp/v1/X.csv $(SCRIPTS_v1) models/v1/feature_models/prior_mean.pt models/v1/feature_models/gpqr.pt
	mkdir -p $(@D)
	$(GPU_PYTHON) -m models.v1.feature_models.predict-gpqr $< --index-col 0 1 2 -o $@

benchmarks/v1/gpqr.Xpred_1D.csv: _temp/v1/Xpred_1D.csv $(SCRIPTS_v1) models/v1/feature_models/prior_mean.pt models/v1/feature_models/gpqr.pt
	mkdir -p $(@D)
	$(GPU_PYTHON) -m models.v1.feature_models.predict-gpqr $< --index-col 0 1 2 -o $@

benchmarks/v1/gpqr.Xpred_2D.csv: _temp/v1/Xpred_2D.csv $(SCRIPTS_v1) models/v1/feature_models/prior_mean.pt models/v1/feature_models/gpqr.pt
	mkdir -p $(@D)
	$(GPU_PYTHON) -m models.v1.feature_models.predict-gpqr $< --index-col 0 1 2 -o $@

benchmarks/v1/gpqr_%.Xtest.csv: _temp/v1/Xtest.csv _temp/v1/prior_mean.pt _temp/v1/gpqr_%.pt $(SCRIPTS_v1)
	mkdir -p $(@D)
	$(GPU_PYTHON) -m models.v1.feature_models.predict-gpqr $(wordlist 1,3,$^) --index-col 0 --batch-col 0 -o $@

# Model selection

benchmarks/v1/pinball_loss.gpr.csv: scripts/v0/model_selection/pinball_loss.py benchmarks/v1/gpr.Xtest.csv _temp/v1/ytest.csv
	mkdir -p $(@D)
	python3 $^ --type GPR --quantile-levels $(QUANTILES) -o $@

benchmarks/v1/pinball_loss.gpqr_%.csv: scripts/v0/model_selection/pinball_loss.py benchmarks/v1/gpqr_%.Xtest.csv _temp/v1/ytest.csv
	mkdir -p $(@D)
	python3 $^ --type GPQR --quantile-levels $(QUANTILES) -o $@

# Window prediction

benchmarks/v1/pit.csv: scripts/v0/joint/write-pit.py _temp/v1/y.csv benchmarks/v1/gpqr.X.csv
	mkdir -p $(@D)
	$(GPU_PYTHON) $^ --index-col 0 --quantiles $(QUANTILES) -o $@

benchmarks/v1/H.marginal.%.csv: scripts/v0/joint/write-marginal.py benchmarks/v1/gpqr.%.csv
	mkdir -p $(@D)
	$(GPU_PYTHON) $^ --target H --quantiles $(QUANTILES) --threshold $(H_THRESHOLD) -o $@

benchmarks/v1/phi_1.marginal.%.csv: scripts/v0/joint/write-marginal.py benchmarks/v1/gpqr.%.csv
	mkdir -p $(@D)
	$(GPU_PYTHON) $^ --target phi_1 --quantiles $(QUANTILES) --threshold $(PHI_THRESHOLD) -o $@

benchmarks/v1/marginal.%.csv: benchmarks/v1/H.marginal.%.csv benchmarks/v1/phi_1.marginal.%.csv
	mkdir -p $(@D)
	python3 -c "import pandas as pd; frames = [pd.read_csv(f) for f in '$^'.split(' ')]; pd.concat(frames, ignore_index=True).to_csv('$@', index=False)"

benchmarks/v1/joint_probability.Xpred_1D.csv: scripts/v0/joint/write-joint.py _temp/v1/Xpred_1D.csv benchmarks/v1/pit.csv benchmarks/v1/marginal.Xpred_1D.csv
	mkdir -p $(@D)
	$(GPU_PYTHON) $^ --index-col 0 1 2 -o $@

benchmarks/v1/joint_probability.Xpred_2D.csv: scripts/v0/joint/write-joint.py _temp/v1/Xpred_2D.csv benchmarks/v1/pit.csv benchmarks/v1/marginal.Xpred_2D.csv
	mkdir -p $(@D)
	$(GPU_PYTHON) $^ --index-col 0 1 2 -o $@

# Class probability prediction

benchmarks/v1/phi_1.class_marginal.%.csv: scripts/v0/joint/write-marginal.py benchmarks/v1/gpqr.%.csv
	mkdir -p $(@D)
	$(GPU_PYTHON) $^ --target phi_1 --quantiles $(QUANTILES) --threshold 0 -o $@

benchmarks/v1/phi_3.class_marginal.%.csv: scripts/v0/joint/write-marginal.py benchmarks/v1/gpqr.%.csv
	mkdir -p $(@D)
	$(GPU_PYTHON) $^ --target phi_3 --quantiles $(QUANTILES) --threshold 0 -o $@

# Examples

examples/v1/CV.ipynb: \
benchmarks/v1/pinball_loss.gpr.csv \
benchmarks/v1/pinball_loss.gpqr_independent.csv \
benchmarks/v1/pinball_loss.gpqr_lmc.csv \
benchmarks/v1/pinball_loss.gpqr_cglmc.csv \
.FORCE
	$(GPU_JUPYTER) nbconvert --to notebook --execute --inplace $@

examples/v1/Models.prior_mean.ipynb: \
models/v1/feature_models/prior_mean.pt \
_temp/v1/X.csv _temp/v1/y.csv _temp/v1/Xpred_1D.csv \
benchmarks/v1/prior_mean.Xpred_1D.csv \
.FORCE
	$(GPU_JUPYTER) nbconvert --to notebook --execute --inplace $@

examples/v1/Models.gpr.ipynb: \
_temp/v1/X.csv _temp/v1/y.csv _temp/v1/Xpred_1D.csv \
benchmarks/v1/gpr.Xpred_1D.csv \
benchmarks/v1/gpr.Xpred_2D.csv \
.FORCE
	$(GPU_JUPYTER) nbconvert --to notebook --execute --inplace $@

examples/v1/Models.ipynb: \
_temp/v1/X.csv _temp/v1/y.csv _temp/v1/Xpred_1D.csv \
benchmarks/v1/prior_mean.Xpred_1D.csv \
benchmarks/v1/gpr.Xpred_1D.csv \
benchmarks/v1/gpqr.Xpred_1D.csv \
.FORCE
	$(GPU_JUPYTER) nbconvert --to notebook --execute --inplace $@

examples/v1/Window.ipynb: \
_temp/v1/X.csv _temp/v1/Xpred_1D.csv _temp/v1/Xpred_2D.csv _temp/v1/delaunay.Xpred_2D.csv \
benchmarks/v1/H.marginal.Xpred_2D.csv benchmarks/v1/phi_1.marginal.Xpred_2D.csv \
benchmarks/v1/joint_probability.Xpred_2D.csv \
benchmarks/v1/joint_probability.Xpred_1D.csv \
.FORCE
	$(GPU_JUPYTER) nbconvert --to notebook --execute --inplace $@

examples/v1/Class_Probability.ipynb: \
_temp/v1/X.csv _temp/v1/Xpred_2D.csv _temp/v1/delaunay.Xpred_2D.csv \
benchmarks/v1/phi_1.class_marginal.Xpred_2D.csv benchmarks/v1/phi_3.class_marginal.Xpred_2D.csv \
.FORCE
	$(GPU_JUPYTER) nbconvert --to notebook --execute --inplace $@
