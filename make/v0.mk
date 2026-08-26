.PHONY: models-v0 examples-v0 test-v0

N_FOLDS_v0 := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),2,10)
N_EPOCHS_v0 := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),1,10000)
N_TRIALS_v0 := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),1,100)

MODELS_v0 := \
models/v0/feature_models/H.prior_mean.pt \
models/v0/feature_models/phi.prior_mean.pt \
models/v0/feature_models/H.gpr.pt \
models/v0/feature_models/phi.gpr.pt \
models/v0/feature_models/H.gpqr.pt \
models/v0/feature_models/phi.gpqr.pt

SCRIPTS_v0 := \
models/v0/feature_models/prior.py \
models/v0/feature_models/scale.py \
models/v0/feature_models/gpr.py \
models/v0/feature_models/gpqr.py \
models/v0/feature_models/likelihoods.py \
models/v0/feature_models/load.py \
models/v0/feature_models/batch.py \
models/v0/feature_models/predict-prior_mean.py \
models/v0/feature_models/predict-gpr.py \
models/v0/feature_models/predict-gpqr.py

models-v0: $(MODELS_v0) $(SCRIPTS_v0)

examples-v0: $(wildcard examples/v0/*)

test-v0: $(if $(filter 1,$(PREBUILT_MODELS)),,$(MODELS_v0) $(SCRIPTS_v0))
	pytest tests/v0 --models-path $(CURDIR)/models/v0

# Data

_temp/v0/dimless.csv: $(wildcard _data/v1/dimless/mean_profiles/dataset*.csv)
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

_temp/v0/shape_features.csv: $(wildcard _data/v1/shape_features/mean_profiles/dataset*.csv)
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

_temp/v0/X.csv: scripts/v0/data/write-X.py _temp/v0/dimless.csv
	python3 $^ -o $@

_temp/v0/Xunique.csv: scripts/v0/data/write-Xunique.py _temp/v0/X.csv
	python3 $^ --index-col 0 1 -o $@

_temp/v0/y.csv: scripts/v0/data/write-y.py _temp/v0/X.csv _temp/v0/shape_features.csv
	python3 $^ --index-col 0 1 2 -o $@

_temp/v0/Xsplit.csv: scripts/v0/data/split-X.py _temp/v0/X.csv
	python3 $^ --split-ratio 0.8 0.1 0.1 --num-folds $(N_FOLDS_v0) --random-state=42 -o $@

_temp/v0/ysplit.csv: scripts/v0/data/write-y.py _temp/v0/Xsplit.csv _temp/v0/shape_features.csv
	python3 $^ --index-col 0 1 2 3 4 -o $@

define SPLIT_v0
_temp/v0/X$(1).csv: _temp/v0/Xsplit.csv
	python3 -c "import pandas as pd; df = pd.read_csv('$$<'); mask = df['split'] == '$(1)'; df.loc[mask, ['fold', 'gap_to_thickness_ratio', 'capillary_number', 'cosine_of_contact_angle']].to_csv('$$@', index=False)"
_temp/v0/y$(1).csv: _temp/v0/ysplit.csv
	python3 -c "import pandas as pd; df = pd.read_csv('$$<'); mask = df['split'] == '$(1)'; df.loc[mask, ['fold', 'H', 'phi']].to_csv('$$@', index=False)"
endef
$(foreach split,train val test,$(eval $(call SPLIT_v0,$(split))))

_temp/v0/Xpred_1D.csv: scripts/v0/data/write-Xpred.py _temp/v0/Xunique.csv
	python3 $^ --target gap_to_thickness_ratio --ngrid $(N_GRID_1) -o $@

_temp/v0/Xpred_2D.csv: scripts/v0/data/write-Xpred.py _temp/v0/Xunique.csv
	python3 $^ --target gap_to_thickness_ratio capillary_number --ngrid $(N_GRID_1) -o $@

_temp/v0/Xpred_3D.csv: scripts/v0/data/write-Xpred.py _temp/v0/Xunique.csv
	python3 $^ --target gap_to_thickness_ratio capillary_number cosine_of_contact_angle --start=0 --stop=1 --ngrid=$(N_GRID_2) -o $@

_temp/v0/delaunay.Xpred_2D.csv: scripts/v0/data/compute-Delaunay.py _temp/v0/Xunique.csv _temp/v0/Xpred_2D.csv
	python3 $^ --grid gap_to_thickness_ratio capillary_number -o $@

# Models

models/v0/feature_models/%.py: scripts/v0/model/%.py
	mkdir -p $(@D)
	cp $< $@

## Prior mean

_temp/v0/%.prior_mean.pt: scripts/v0/train/prior_mean.py _temp/v0/Xtrain.csv _temp/v0/ytrain.csv \
$(SCRIPTS_v0)
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,3,$^) --index-col 0 --batch-col 0 --target $* --model PriorMean_$* --num-epochs $(N_EPOCHS_v0) -o $@

models/v0/feature_models/%.prior_mean.pt: scripts/v0/train/prior_mean.py _temp/v0/X.csv _temp/v0/y.csv \
$(SCRIPTS_v0)
	mkdir -p $(@D)
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,3,$^) --index-col 0 1 2 --target $* --model PriorMean_$* --num-epochs $(N_EPOCHS_v0) -o $@

## GPR

_temp/v0/%.gpr.pt: scripts/v0/train/gpr.py _temp/v0/Xtrain.csv _temp/v0/ytrain.csv _temp/v0/Xval.csv _temp/v0/yval.csv _temp/v0/%.prior_mean.pt \
$(SCRIPTS_v0)
	mkdir -p benchmarks
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,6,$^) --index-col 0 --batch-col 0 --target $* --model GPR_$* --num-epochs $(N_EPOCHS_v0) --optimize-hyperparameters noise_prior_loc noise_prior_scale lengthscale_prior_loc lengthscale_prior_scale --n-trials=$(N_TRIALS_v0) --storage=$(OPTUNA_DB) --study-name=v0/$*.gpr -o $@

models/v0/feature_models/%.gpr.pt: scripts/v0/train/gpr.py _temp/v0/X.csv _temp/v0/y.csv models/v0/feature_models/%.prior_mean.pt \
_temp/v0/%.gpr.pt $(SCRIPTS_v0)
	mkdir -p $(@D)
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,4,$^) --index-col 0 1 2 --target $* --model GPR_$* --storage=$(OPTUNA_DB) --study-name=v0/$*.gpr -o $@

## GPQR

_temp/v0/%.direct_gpqr_independent.pt: scripts/v0/train/gpqr.py _temp/v0/Xtrain.csv _temp/v0/ytrain.csv _temp/v0/Xval.csv _temp/v0/yval.csv _temp/v0/%.prior_mean.pt \
$(SCRIPTS_v0)
	mkdir -p benchmarks
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,6,$^) --index-col 0 --batch-col 0 --target $* --model DirectMTGPQR_Independent_$* --quantiles $(QUANTILES) --num-likelihood-samples $(N_LIKELIHOOD_SAMPLES) --num-epochs $(N_EPOCHS_v0) --optimize-hyperparameters noise_prior_loc noise_prior_scale lengthscale_prior_loc lengthscale_prior_scale --n-trials=$(N_TRIALS_v0) --storage=$(OPTUNA_DB) --study-name=v0/$*.direct_gpqr_independent -o $@

_temp/v0/%.direct_gpqr_lmc.pt: scripts/v0/train/gpqr.py _temp/v0/Xtrain.csv _temp/v0/ytrain.csv _temp/v0/Xval.csv _temp/v0/yval.csv _temp/v0/%.prior_mean.pt \
$(SCRIPTS_v0)
	mkdir -p benchmarks
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,6,$^) --index-col 0 --batch-col 0 --target $* --model DirectMTGPQR_LMC_$* --quantiles $(QUANTILES) --num-likelihood-samples $(N_LIKELIHOOD_SAMPLES) --num-epochs $(N_EPOCHS_v0) --optimize-hyperparameters noise_prior_loc noise_prior_scale lengthscale_prior_loc lengthscale_prior_scale num_latents --n-trials=$(N_TRIALS_v0) --storage=$(OPTUNA_DB) --study-name=v0/$*.direct_gpqr_lmc -o $@

_temp/v0/%.cg_gpqr_independent.pt: scripts/v0/train/gpqr.py _temp/v0/Xtrain.csv _temp/v0/ytrain.csv _temp/v0/Xval.csv _temp/v0/yval.csv _temp/v0/%.prior_mean.pt \
$(SCRIPTS_v0)
	mkdir -p benchmarks
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,6,$^) --index-col 0 --batch-col 0 --target $* --model CenterGapMTGPQR_Independent_$* --quantiles $(QUANTILES) --num-likelihood-samples $(N_LIKELIHOOD_SAMPLES) --num-epochs $(N_EPOCHS_v0) --optimize-hyperparameters noise_prior_loc noise_prior_scale lengthscale_prior_loc lengthscale_prior_scale --n-trials=$(N_TRIALS_v0) --storage=$(OPTUNA_DB) --study-name=v0/$*.cg_gpqr_independent -o $@

_temp/v0/%.cg_gpqr_lmc.pt: scripts/v0/train/gpqr.py _temp/v0/Xtrain.csv _temp/v0/ytrain.csv _temp/v0/Xval.csv _temp/v0/yval.csv _temp/v0/%.prior_mean.pt \
$(SCRIPTS_v0)
	mkdir -p benchmarks
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,6,$^) --index-col 0 --batch-col 0 --target $* --model CenterGapMTGPQR_LMC_$* --quantiles $(QUANTILES) --num-likelihood-samples $(N_LIKELIHOOD_SAMPLES) --num-epochs $(N_EPOCHS_v0) --optimize-hyperparameters noise_prior_loc noise_prior_scale lengthscale_prior_loc lengthscale_prior_scale num_latents --n-trials=$(N_TRIALS_v0) --storage=$(OPTUNA_DB) --study-name=v0/$*.cg_gpqr_lmc -o $@

_temp/v0/%.cg_gpqr_cglmc.pt: scripts/v0/train/gpqr.py _temp/v0/Xtrain.csv _temp/v0/ytrain.csv _temp/v0/Xval.csv _temp/v0/yval.csv _temp/v0/%.prior_mean.pt \
$(SCRIPTS_v0)
	mkdir -p benchmarks
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,6,$^) --index-col 0 --batch-col 0 --target $* --model CenterGapMTGPQR_CenterGapLMC_$* --quantiles $(QUANTILES) --num-likelihood-samples $(N_LIKELIHOOD_SAMPLES) --num-epochs $(N_EPOCHS_v0) --optimize-hyperparameters noise_prior_loc noise_prior_scale lengthscale_prior_loc lengthscale_prior_scale num_latents --n-trials=$(N_TRIALS_v0) --storage=$(OPTUNA_DB) --study-name=v0/$*.cg_gpqr_cglmc -o $@

models/v0/feature_models/%.gpqr.pt: scripts/v0/train/gpqr.py _temp/v0/X.csv _temp/v0/y.csv models/v0/feature_models/%.prior_mean.pt \
_temp/v0/%.cg_gpqr_independent.pt $(SCRIPTS_v0)
	mkdir -p $(@D)
	PYTHONPATH=. $(GPU_PYTHON) $(wordlist 1,4,$^) --index-col 0 1 2 --target $* --model CenterGapMTGPQR_Independent_$* --quantiles $(QUANTILES) --num-likelihood-samples $(N_LIKELIHOOD_SAMPLES) --storage=$(OPTUNA_DB) --study-name=v0/$*.cg_gpqr_independent -o $@

# Prediction

_temp/v0/%.prior_mean.Xpred_1D.csv: _temp/v0/Xpred_1D.csv $(SCRIPTS_v0) models/v0/feature_models/%.prior_mean.pt
	$(GPU_PYTHON) -m models.v0.feature_models.predict-prior_mean $< --index-col 0 1 2 --target $* -o $@

_temp/v0/%.gpr.Xpred_1D.csv: _temp/v0/Xpred_1D.csv $(SCRIPTS_v0) models/v0/feature_models/%.prior_mean.pt models/v0/feature_models/%.gpr.pt
	$(GPU_PYTHON) -m models.v0.feature_models.predict-gpr $< --index-col 0 1 2 --target $* -o $@

_temp/v0/%.gpr.Xtest.csv: _temp/v0/Xtest.csv _temp/v0/%.prior_mean.pt _temp/v0/%.gpr.pt $(SCRIPTS_v0)
	$(GPU_PYTHON) -m models.v0.feature_models.predict-gpr $(wordlist 1,3,$^) --index-col 0 --batch-col 0 --target $* -o $@

_temp/v0/%.gpqr.X.csv: _temp/v0/X.csv $(SCRIPTS_v0) models/v0/feature_models/%.prior_mean.pt models/v0/feature_models/%.gpqr.pt
	$(GPU_PYTHON) -m models.v0.feature_models.predict-gpqr $< --index-col 0 1 2 --target $* -o $@

_temp/v0/%.gpqr.Xpred_1D.csv: _temp/v0/Xpred_1D.csv $(SCRIPTS_v0) models/v0/feature_models/%.prior_mean.pt models/v0/feature_models/%.gpqr.pt
	$(GPU_PYTHON) -m models.v0.feature_models.predict-gpqr $< --index-col 0 1 2 --target $* -o $@

_temp/v0/%.gpqr.Xpred_2D.csv: _temp/v0/Xpred_2D.csv $(SCRIPTS_v0) models/v0/feature_models/%.prior_mean.pt models/v0/feature_models/%.gpqr.pt
	$(GPU_PYTHON) -m models.v0.feature_models.predict-gpqr $< --index-col 0 1 2 --target $* -o $@

_temp/v0/%.direct_gpqr_independent.Xpred_3D.csv: _temp/v0/Xpred_3D.csv _temp/v0/%.prior_mean.pt _temp/v0/%.direct_gpqr_independent.pt $(SCRIPTS_v0)
	$(GPU_PYTHON) -m models.v0.feature_models.predict-gpqr $(wordlist 1,3,$^) --index-col 0 1 2 --target $* -o $@

_temp/v0/%.direct_gpqr_lmc.Xpred_3D.csv: _temp/v0/Xpred_3D.csv _temp/v0/%.prior_mean.pt _temp/v0/%.direct_gpqr_lmc.pt $(SCRIPTS_v0)
	$(GPU_PYTHON) -m models.v0.feature_models.predict-gpqr $(wordlist 1,3,$^) --index-col 0 1 2 --target $* -o $@

_temp/v0/%.cg_gpqr_independent.Xtest.csv: _temp/v0/Xtest.csv _temp/v0/%.prior_mean.pt _temp/v0/%.cg_gpqr_independent.pt $(SCRIPTS_v0)
	$(GPU_PYTHON) -m models.v0.feature_models.predict-gpqr $(wordlist 1,3,$^) --index-col 0 --batch-col 0 --target $* -o $@

_temp/v0/%.cg_gpqr_lmc.Xtest.csv: _temp/v0/Xtest.csv _temp/v0/%.prior_mean.pt _temp/v0/%.cg_gpqr_lmc.pt $(SCRIPTS_v0)
	$(GPU_PYTHON) -m models.v0.feature_models.predict-gpqr $(wordlist 1,3,$^) --index-col 0 --batch-col 0 --target $* -o $@

_temp/v0/%.cg_gpqr_cglmc.Xtest.csv: _temp/v0/Xtest.csv _temp/v0/%.prior_mean.pt _temp/v0/%.cg_gpqr_cglmc.pt $(SCRIPTS_v0)
	$(GPU_PYTHON) -m models.v0.feature_models.predict-gpqr $(wordlist 1,3,$^) --index-col 0 --batch-col 0 --target $* -o $@

# Model selection

benchmarks/v0/quantile_crossing.%.direct_gpqr_independent.csv: scripts/v0/model_selection/crossing.py _temp/v0/%.direct_gpqr_independent.Xpred_3D.csv
	mkdir -p $(@D)
	python3 $^ -o $@

benchmarks/v0/quantile_crossing.%.direct_gpqr_lmc.csv: scripts/v0/model_selection/crossing.py _temp/v0/%.direct_gpqr_lmc.Xpred_3D.csv
	mkdir -p $(@D)
	python3 $^ -o $@

benchmarks/v0/pinball_loss.%.gpr.csv: scripts/v0/model_selection/pinball_loss.py _temp/v0/%.gpr.Xtest.csv _temp/v0/ytest.csv
	mkdir -p $(@D)
	python3 $^ --type GPR --quantile-levels $(QUANTILES) -o $@

benchmarks/v0/pinball_loss.%.cg_gpqr_independent.csv: scripts/v0/model_selection/pinball_loss.py _temp/v0/%.cg_gpqr_independent.Xtest.csv _temp/v0/ytest.csv
	mkdir -p $(@D)
	python3 $^ --type GPQR --quantile-levels $(QUANTILES) -o $@

benchmarks/v0/pinball_loss.%.cg_gpqr_lmc.csv: scripts/v0/model_selection/pinball_loss.py _temp/v0/%.cg_gpqr_lmc.Xtest.csv _temp/v0/ytest.csv
	mkdir -p $(@D)
	python3 $^ --type GPQR --quantile-levels $(QUANTILES) -o $@

benchmarks/v0/pinball_loss.%.cg_gpqr_cglmc.csv: scripts/v0/model_selection/pinball_loss.py _temp/v0/%.cg_gpqr_cglmc.Xtest.csv _temp/v0/ytest.csv
	mkdir -p $(@D)
	python3 $^ --type GPQR --quantile-levels $(QUANTILES) -o $@

# Window prediction

_temp/v0/%.pit.csv: scripts/v0/joint/write-pit.py _temp/v0/y.csv _temp/v0/%.gpqr.X.csv
	python3 $^ --index-col 0 --quantiles $(QUANTILES) -o $@

_temp/v0/H.marginal.Xpred_2D.csv: scripts/v0/joint/write-marginal.py _temp/v0/H.gpqr.Xpred_2D.csv
	python3 $^ --quantiles $(QUANTILES) --threshold $(H_THRESHOLD) -o $@

_temp/v0/phi.marginal.Xpred_2D.csv: scripts/v0/joint/write-marginal.py _temp/v0/phi.gpqr.Xpred_2D.csv
	python3 $^ --quantiles $(QUANTILES) --threshold $(PHI_THRESHOLD) -o $@

_temp/v0/H_phi.pit.csv: _temp/v0/H.pit.csv _temp/v0/phi.pit.csv
	python3 -c "import pandas as pd; H, phi = map(lambda f: pd.read_csv(f), '$^'.split(' ')); pd.concat([H, phi], ignore_index=True).to_csv('$@', index=False)"

_temp/v0/H_phi.marginal.Xpred_2D.csv: _temp/v0/H.marginal.Xpred_2D.csv _temp/v0/phi.marginal.Xpred_2D.csv
	python3 -c "import pandas as pd; H, phi = map(lambda f: pd.read_csv(f), '$^'.split(' ')); pd.concat([H, phi], ignore_index=True).to_csv('$@', index=False)"

_temp/v0/joint_probability.Xpred_2D.csv: scripts/v0/joint/write-joint.py _temp/v0/Xpred_2D.csv _temp/v0/H_phi.pit.csv _temp/v0/H_phi.marginal.Xpred_2D.csv
	python3 $^ --index-col 0 1 2 -o $@

# Examples

examples/v0/Crossing.ipynb: benchmarks/v0/quantile_crossing.H.direct_gpqr_independent.csv benchmarks/v0/quantile_crossing.phi.direct_gpqr_independent.csv benchmarks/v0/quantile_crossing.H.direct_gpqr_lmc.csv benchmarks/v0/quantile_crossing.phi.direct_gpqr_lmc.csv .FORCE
	$(GPU_JUPYTER) nbconvert --to notebook --execute --inplace $@

examples/v0/CV.ipynb: \
benchmarks/v0/pinball_loss.H.gpr.csv benchmarks/v0/pinball_loss.phi.gpr.csv \
benchmarks/v0/pinball_loss.H.cg_gpqr_independent.csv benchmarks/v0/pinball_loss.phi.cg_gpqr_independent.csv \
benchmarks/v0/pinball_loss.H.cg_gpqr_lmc.csv benchmarks/v0/pinball_loss.phi.cg_gpqr_lmc.csv \
benchmarks/v0/pinball_loss.H.cg_gpqr_cglmc.csv benchmarks/v0/pinball_loss.phi.cg_gpqr_cglmc.csv \
.FORCE
	$(GPU_JUPYTER) nbconvert --to notebook --execute --inplace $@

examples/v0/Models.ipynb: \
_temp/v0/X.csv _temp/v0/y.csv _temp/v0/Xpred_1D.csv \
_temp/v0/H.prior_mean.Xpred_1D.csv _temp/v0/phi.prior_mean.Xpred_1D.csv \
_temp/v0/H.gpr.Xpred_1D.csv _temp/v0/phi.gpr.Xpred_1D.csv \
_temp/v0/H.gpqr.Xpred_1D.csv _temp/v0/phi.gpqr.Xpred_1D.csv \
.FORCE
	$(GPU_JUPYTER) nbconvert --to notebook --execute --inplace $@

examples/v0/Window.ipynb: \
_temp/v0/X.csv _temp/v0/Xpred_2D.csv _temp/v0/delaunay.Xpred_2D.csv \
_temp/v0/H.marginal.Xpred_2D.csv _temp/v0/phi.marginal.Xpred_2D.csv \
_temp/v0/joint_probability.Xpred_2D.csv \
.FORCE
	$(GPU_JUPYTER) nbconvert --to notebook --execute --inplace $@
