.PHONY: models-v0 examples-v0 test-v0

QUANTILES := 0.05 0.25 0.5 0.75 0.95
NUM_LOWER_QUANTILES := 2
NUM_LATENTS := 3

N_EPOCHS := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),1,10000)
N_FOLDS := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),2,10)
N_GRID_1 := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),2,200)
N_GRID_2 := $(if $(filter 1,$(HEAVYEDGE_TEST_MODE)),2,10)

H_THRESHOLD := 1.1
PHI_THRESHOLD := 1.0

PYTHON ?= python3
GPU_RUN ?= $(PYTHON) scripts/gpu-run.py --
GPU_PYTHON ?= $(GPU_RUN) $(PYTHON)
GPU_JUPYTER ?= $(GPU_RUN) jupyter

MODEL_FILES := \
models/v0/feature_models/H.prior_mean.pt \
models/v0/feature_models/phi.prior_mean.pt \
models/v0/feature_models/H.gpr.pt \
models/v0/feature_models/phi.gpr.pt \
models/v0/feature_models/H.gpqr.pt \
models/v0/feature_models/phi.gpqr.pt \
models/v0/prior.py \
models/v0/setup.py \
models/v0/scale.py \
models/v0/gpr.py \
models/v0/gpqr.py \
models/v0/load.py \
models/v0/predict-prior_mean.py \
models/v0/predict-mean.py \
models/v0/predict-quantiles.py

models-v0: $(MODEL_FILES)

examples-v0: $(wildcard examples/v0/*)

test-v0:
	$(GPU_PYTHON) -c "from models.v0.load import load_PriorMean_H; load_PriorMean_H()"
	$(GPU_PYTHON) -c "from models.v0.load import load_PriorMean_phi; load_PriorMean_phi()"
	$(GPU_PYTHON) -c "from models.v0.load import load_GPR_H; load_GPR_H()"
	$(GPU_PYTHON) -c "from models.v0.load import load_GPR_phi; load_GPR_phi()"
	$(GPU_PYTHON) -c "from models.v0.load import load_GPQR_H; load_GPQR_H()"
	$(GPU_PYTHON) -c "from models.v0.load import load_GPQR_phi; load_GPQR_phi()"

# Data

_temp/v0/X.csv: scripts/v0/data/write-X.py _data/v1/dimless.csv
	mkdir -p $(@D)
	python3 $^ -o $@

_temp/v0/y.csv: scripts/v0/data/write-y.py _temp/v0/X.csv _data/v1/shape_features/mean_profiles.csv
	python3 $^ -o $@

_temp/v0/Xtrain.csv: _temp/v0/X.csv
	python3 -c "import pandas as pd; pd.read_csv('$<')[['gap_to_thickness_ratio', 'capillary_number', 'cosine_of_contact_angle']].to_csv('$@', index=False)"

_temp/v0/ytrain.csv: _temp/v0/y.csv
	python3 -c "import pandas as pd; pd.read_csv('$<')[['H', 'phi']].to_csv('$@', index=False)"

_temp/v0/Xpred.csv: _temp/v0/Xtrain.csv
	python3 -c "import pandas as pd; X = pd.read_csv('$<'); cols = ['gap_to_thickness_ratio', 'capillary_number', 'cosine_of_contact_angle']; idx = pd.DataFrame({col + '_idx': X[col].map({value: i for i, value in enumerate(sorted(X[col].unique()))}) for col in cols}); pd.concat([idx, X], axis=1).to_csv('$@', index=False)"

_temp/v0/Xpred_1D.csv: scripts/v0/data/write-Xpred.py _temp/v0/Xtrain.csv
	python3 $^ --target gap_to_thickness_ratio --ngrid $(N_GRID_1) -o $@

_temp/v0/Xpred_2D.csv: scripts/v0/data/write-Xpred.py _temp/v0/Xtrain.csv
	python3 $^ --target gap_to_thickness_ratio capillary_number --ngrid $(N_GRID_1) -o $@

_temp/v0/Xpred_3D-1.csv: scripts/v0/data/write-Xpred.py _temp/v0/Xtrain.csv
	python3 $^ --target gap_to_thickness_ratio capillary_number cosine_of_contact_angle --start=0 --stop=1 --ngrid=$(N_GRID_2) -o $@

_temp/v0/Xpred_3D-2.csv: scripts/v0/data/write-Xpred.py _temp/v0/Xtrain.csv
	python3 $^ --target gap_to_thickness_ratio capillary_number cosine_of_contact_angle --start=-2 --stop=2 --ngrid=$(N_GRID_2) -o $@

# Prior mean

# ex) models/v0/feature_models/H.prior_mean.pt
models/v0/feature_models/%.prior_mean.pt: scripts/v0/train/prior_mean.py _temp/v0/Xtrain.csv _temp/v0/ytrain.csv
	mkdir -p $(@D)
	PYTHONPATH=scripts/v0 $(GPU_PYTHON) $(wordlist 1,3,$^) --target $* --model PriorMean_$* --num-epochs $(N_EPOCHS) -o $@

# Model selection

_temp/v0/crossing.DirectMTGPQR_%.csv: scripts/v0/model_selection/write-crossing.py _temp/v0/Xtrain.csv _temp/v0/ytrain.csv models/v0/feature_models/%.prior_mean.pt _temp/v0/Xpred_3D-1.csv _temp/v0/Xpred_3D-2.csv
	PYTHONPATH=scripts/v0 $(GPU_PYTHON) $^ --target $* --model DirectMTGPQR_$* --quantiles $(QUANTILES) --num-latents $(NUM_LATENTS) --n-epochs $(N_EPOCHS) -o $@

_temp/v0/extrapolation.CenterGapMTGPQR_%.csv: scripts/v0/model_selection/write-extrapolation.py _temp/v0/Xtrain.csv _temp/v0/ytrain.csv models/v0/feature_models/%.prior_mean.pt
	PYTHONPATH=scripts/v0 $(GPU_PYTHON) $^ --target $* --model CenterGapMTGPQR_$* --split-ratio=0.5 --quantiles $(QUANTILES) --num-lower-quantiles $(NUM_LOWER_QUANTILES) --num-latents $(NUM_LATENTS) --n-epochs $(N_EPOCHS) -o $@

_temp/v0/extrapolation.CenterGapMTGPQR_%_ConstantMean.csv: scripts/v0/model_selection/write-extrapolation.py _temp/v0/Xtrain.csv _temp/v0/ytrain.csv
	PYTHONPATH=scripts/v0 $(GPU_PYTHON) $^ --target $* --model CenterGapMTGPQR_$* --split-ratio=0.5 --quantiles $(QUANTILES) --num-lower-quantiles $(NUM_LOWER_QUANTILES) --num-latents $(NUM_LATENTS) --n-epochs $(N_EPOCHS) -o $@

_temp/v0/cv.GPR_%.csv: scripts/v0/model_selection/write-cv.gpr.py _temp/v0/Xtrain.csv _temp/v0/ytrain.csv models/v0/feature_models/%.prior_mean.pt
	PYTHONPATH=scripts/v0 $(GPU_PYTHON) $^ --target $* --model GPR_$* --num-folds=$(N_FOLDS) --quantiles $(QUANTILES) --n-epochs $(N_EPOCHS) -o $@

_temp/v0/cv.CenterGapMTGPQR_%.csv: scripts/v0/model_selection/write-cv.gpqr.py _temp/v0/Xtrain.csv _temp/v0/ytrain.csv models/v0/feature_models/%.prior_mean.pt
	PYTHONPATH=scripts/v0 $(GPU_PYTHON) $^ --target $* --model CenterGapMTGPQR_$* --num-folds=$(N_FOLDS) --quantiles $(QUANTILES) --num-lower-quantiles $(NUM_LOWER_QUANTILES) --num-latents $(NUM_LATENTS) --n-epochs $(N_EPOCHS) -o $@

# Model

_temp/v0/best-config.%.mean.epoch: scripts/v0/train/write-best.py _temp/v0/cv.GPR_%.csv
	python3 $^ --target epoch -o $@ 0

models/v0/feature_models/%.gpr.pt: scripts/v0/train/gpr.py _temp/v0/Xtrain.csv _temp/v0/ytrain.csv models/v0/feature_models/%.prior_mean.pt _temp/v0/best-config.%.mean.epoch
	mkdir -p $(@D)
	PYTHONPATH=scripts/v0 $(GPU_PYTHON) $(wordlist 1,4,$^) --target $* --model GPR_$* --num-epochs $(shell cat $(lastword $^)) -o $@

_temp/v0/best-config.%.quantiles.epoch: scripts/v0/train/write-best.py _temp/v0/cv.CenterGapMTGPQR_%.csv
	python3 $^ --target epoch -o $@ 0

models/v0/feature_models/%.gpqr.pt: scripts/v0/train/gpqr.py _temp/v0/Xtrain.csv _temp/v0/ytrain.csv models/v0/feature_models/%.prior_mean.pt _temp/v0/best-config.%.quantiles.epoch
	mkdir -p $(@D)
	PYTHONPATH=scripts/v0 $(GPU_PYTHON) $(wordlist 1,4,$^) --target $* --model CenterGapMTGPQR_$* --quantiles $(QUANTILES) --num-lower-quantiles $(NUM_LOWER_QUANTILES) --num-latents $(NUM_LATENTS) --num-epochs $(shell cat $(lastword $^)) -o $@

models/v0/%.py: scripts/v0/model/%.py
	mkdir -p $(@D)
	cp $< $@


# Prediction

_temp/v0/%.prior_mean.Xpred_1D.csv: models/v0/predict-prior_mean.py _temp/v0/Xpred_1D.csv $(MODEL_FILES)
	$(GPU_PYTHON) $(wordlist 1,2,$^) --target $* -o $@

_temp/v0/%.prior_mean.Xpred_2D.csv: models/v0/predict-prior_mean.py _temp/v0/Xpred_2D.csv $(MODEL_FILES)
	$(GPU_PYTHON) $(wordlist 1,2,$^) --target $* -o $@

_temp/v0/%.mean.Xpred_1D.csv: models/v0/predict-mean.py _temp/v0/Xpred_1D.csv $(MODEL_FILES)
	$(GPU_PYTHON) $(wordlist 1,2,$^) --target $* -o $@

_temp/v0/%.quantiles.X.csv: models/v0/predict-quantiles.py _temp/v0/Xpred.csv $(MODEL_FILES)
	$(GPU_PYTHON) $(wordlist 1,2,$^) --target $* --method delta -o $@

_temp/v0/%.quantiles.Xpred_1D.csv: models/v0/predict-quantiles.py _temp/v0/Xpred_1D.csv $(MODEL_FILES)
	$(GPU_PYTHON) $(wordlist 1,2,$^) --target $* --method delta -o $@

_temp/v0/%.quantiles.Xpred_2D.csv: models/v0/predict-quantiles.py _temp/v0/Xpred_2D.csv $(MODEL_FILES)
	$(GPU_PYTHON) $(wordlist 1,2,$^) --target $* --method delta -o $@

# Window prediction

_temp/delaunay.Xpred_2D.npy: scripts/data/compute-Delaunay.py _temp/X.csv _temp/Xpred_2D.csv
	python3 $^ --grid Gap_to_thickness_ratio Capillary_number -o $@

_temp/%.pit.Xpred_2D.npy: scripts/joint/write-pit.py _temp/y.csv _temp/%.quantiles.X.npy
	python3 $^ --target $* --quantiles $(QUANTILES) -o $@

_temp/H.marginal.Xpred_2D.npy: scripts/joint/write-marginal.py _temp/H.quantiles.Xpred_2D.npy
	python3 $^ --quantiles $(QUANTILES) --threshold $(H_THRESHOLD) -o $@

_temp/phi.marginal.Xpred_2D.npy: scripts/joint/write-marginal.py _temp/phi.quantiles.Xpred_2D.npy
	python3 $^ --quantiles $(QUANTILES) --threshold $(PHI_THRESHOLD) -o $@

_temp/%.pit_marginal.Xpred_2D.npz: _temp/%.pit.Xpred_2D.npy _temp/%.marginal.Xpred_2D.npy
	python3 -c "import numpy as np; pit, marginal = map(np.load, '$^'.split(' ')); np.savez('$@', pit=pit, marginal=marginal)"

_temp/joint_probability.Xpred_2D.npy: scripts/joint/write-joint.py _temp/Xpred_2D.csv _temp/H.pit_marginal.Xpred_2D.npz _temp/phi.pit_marginal.Xpred_2D.npz
	python3 $^ -o $@

# Examples

examples/Crossing.%.ipynb: _temp/X.csv _temp/y.csv _temp/crossing.DirectMTGPQR_%.csv .FORCE
	$(GPU_JUPYTER) nbconvert --to notebook --execute --inplace $@

examples/Extrapolation.%.ipynb: _temp/X.csv _temp/y.csv _temp/extrapolation.CenterGapMTGPQR_%.npy _temp/extrapolation.CenterGapMTGPQR_%_ConstantMean.npy .FORCE
	$(GPU_JUPYTER) nbconvert --to notebook --execute --inplace $@

examples/CV.%.ipynb: _temp/X.csv _temp/y.csv _temp/cv.GPR_%.npy _temp/cv.CenterGapMTGPQR_%.npy .FORCE
	$(GPU_JUPYTER) nbconvert --to notebook --execute --inplace $@

examples/Models.ipynb: _temp/X.csv _temp/y.csv _temp/Xpred_1D.csv _temp/H.prior_mean.Xpred_1D.npy _temp/phi.prior_mean.Xpred_1D.npy _temp/H.mean.Xpred_1D.npy _temp/phi.mean.Xpred_1D.npy _temp/H.quantiles.Xpred_1D.npy _temp/phi.quantiles.Xpred_1D.npy .FORCE
	$(GPU_JUPYTER) nbconvert --to notebook --execute --inplace $@

examples/Window.ipynb: _temp/X.csv _temp/y.csv _temp/Xpred_2D.csv _temp/delaunay.Xpred_2D.npy _temp/H.prior_mean.Xpred_2D.npy _temp/phi.prior_mean.Xpred_2D.npy _temp/H.quantiles.Xpred_2D.npy _temp/phi.quantiles.Xpred_2D.npy _temp/H.marginal.Xpred_2D.npy _temp/phi.marginal.Xpred_2D.npy _temp/joint_probability.Xpred_2D.npy .FORCE
	$(GPU_JUPYTER) nbconvert --to notebook --execute --inplace $@
