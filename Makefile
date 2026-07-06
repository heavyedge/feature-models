QUANTILES := 0.05 0.25 0.5 0.75 0.95
NUM_LOWER_QUANTILES := 2
NUM_LATENTS := 3
HEAVYEDGE_N_EPOCHS ?= 10000

NOTEBOOKS := $(wildcard notebooks/*)
MODEL_FILES := \
model/H.gpqr.pt \
model/phi.gpqr.pt \
model/prior.py \
model/scale.py \
model/gpqr.py \
model/load.py \
model/predict-mean.py \
model/predict-quantiles.py \
model/requirements.txt

.SECONDARY:
.ONESHELL:
.PHONY: models notebooks test all clean FORCE

models: $(MODEL_FILES)

notebooks: $(NOTEBOOKS)

test:
	python3 -c "from model.load import load_H_models; load_H_models()"
	python3 -c "from model.load import load_phi_models; load_phi_models()"

all: models notebooks

clean:
	rm -rf _temp _artifacts model/*.pt model/*.py model/requirements.txt

# Notebooks

notebooks/Crossing.%.ipynb: _temp/X.csv _temp/y.csv _temp/crossing.DirectMTGPQR_%.csv FORCE
	jupyter nbconvert --to notebook --execute --inplace $@

notebooks/Extrapolation.%.ipynb: _temp/X.csv _temp/y.csv _temp/extrapolation.CenterGapMTGPQR_%.csv _temp/extrapolation.CenterGapMTGPQR_%_ConstantMean.csv FORCE
	jupyter nbconvert --to notebook --execute --inplace $@

notebooks/CV.%.ipynb: _temp/X.csv _temp/y.csv _temp/cv.GPR_%.csv _temp/cv.CenterGapMTGPQR_%.csv FORCE
	jupyter nbconvert --to notebook --execute --inplace $@

notebooks/Quantiles.ipynb: _temp/X.csv _temp/y.csv _temp/Xpred_1D.csv _temp/H.prior_mean.Xpred_1D.npy _temp/phi.prior_mean.Xpred_1D.npy _temp/H.quantiles.Xpred_1D.npy _temp/phi.quantiles.Xpred_1D.npy FORCE
	jupyter nbconvert --to notebook --execute --inplace $@

notebooks/Window.ipynb: _temp/X.csv _temp/y.csv _temp/Xpred_2D.csv _temp/delaunay.Xpred_2D.npy _temp/H.prior_mean.Xpred_2D.npy _temp/phi.prior_mean.Xpred_2D.npy _temp/H.quantiles.Xpred_2D.npy _temp/phi.quantiles.Xpred_2D.npy _temp/H.marginal.Xpred_2D.npy _temp/phi.marginal.Xpred_2D.npy _temp/joint_probability.Xpred_2D.npy FORCE
	jupyter nbconvert --to notebook --execute --inplace $@

FORCE:  # dummy target to force execution of dependent targets

# Data

_temp/Dataset.csv: scripts/data/filter-dataset.py _data/Dataset.csv
	mkdir -p $(@D)
	python3 $^ -o $@

_temp/X.csv: scripts/data/write-X.py _temp/Dataset.csv
	python3 $^ -o $@

_temp/y.csv: _temp/Dataset.csv
	python3 -c "import pandas as pd; pd.read_csv('$<')[['H', 'phi']].to_csv('$@', index=False)"

_temp/Xpred_1D.csv: scripts/data/write-Xpred.py _temp/X.csv
	python3 $^ --target Gap_to_thickness_ratio -o $@

_temp/Xpred_2D.csv: scripts/data/write-Xpred.py _temp/X.csv
	python3 $^ --target Gap_to_thickness_ratio Capillary_number -o $@

_temp/Xpred_3D-1.csv: scripts/data/write-Xpred.py _temp/X.csv
	python3 $^ --target Gap_to_thickness_ratio Capillary_number Cos_theta --start=0 --stop=1 --ngrid=10 -o $@

_temp/Xpred_3D-2.csv: scripts/data/write-Xpred.py _temp/X.csv
	python3 $^ --target Gap_to_thickness_ratio Capillary_number Cos_theta --start=-2 --stop=2 --ngrid=10 -o $@

_temp/X.npy: _temp/X.csv
	python3 -c "import pandas as pd; import numpy as np; np.save('$@', pd.read_csv('$<').drop(columns=['Slurry']).to_numpy())"

_temp/Xpred_1D.npy: _temp/Xpred_1D.csv
	python3 -c "import pandas as pd; import numpy as np; np.save('$@', pd.read_csv('$<', index_col=[0, 1, 2, 3]).to_numpy())"

_temp/Xpred_2D.npy: _temp/Xpred_2D.csv
	python3 -c "import pandas as pd; import numpy as np; np.save('$@', pd.read_csv('$<', index_col=[0, 1, 2, 3]).to_numpy())"

# Model selection

_temp/crossing.DirectMTGPQR_%.csv: scripts/model_selection/write-crossing.py _temp/X.csv _temp/y.csv _temp/%.prior_mean.pt _temp/Xpred_3D-1.csv _temp/Xpred_3D-2.csv
	python3 $^ --target $* --model DirectMTGPQR_$* --quantiles $(QUANTILES) --num-latents $(NUM_LATENTS) --n-epochs $(HEAVYEDGE_N_EPOCHS) -o $@

_temp/extrapolation.CenterGapMTGPQR_%.csv: scripts/model_selection/write-extrapolation.py _temp/X.csv _temp/y.csv _temp/%.prior_mean.pt
	python3 $^ --target $* --model CenterGapMTGPQR_$* --split-ratio=0.5 --quantiles $(QUANTILES) --num-lower-quantiles $(NUM_LOWER_QUANTILES) --num-latents $(NUM_LATENTS) --n-epochs $(HEAVYEDGE_N_EPOCHS) -o $@

_temp/extrapolation.CenterGapMTGPQR_%_ConstantMean.csv: scripts/model_selection/write-extrapolation.py _temp/X.csv _temp/y.csv
	python3 $^ --target $* --model CenterGapMTGPQR_$* --split-ratio=0.5 --quantiles $(QUANTILES) --num-lower-quantiles $(NUM_LOWER_QUANTILES) --num-latents $(NUM_LATENTS) --n-epochs $(HEAVYEDGE_N_EPOCHS) -o $@

_temp/cv.GPR_%.csv: scripts/model_selection/write-cv.gpr.py _temp/X.csv _temp/y.csv _temp/%.prior_mean.pt
	python3 $^ --target $* --model GPR_$* --num-folds=10 --quantiles $(QUANTILES) --n-epochs $(HEAVYEDGE_N_EPOCHS) -o $@

_temp/cv.CenterGapMTGPQR_%.csv: scripts/model_selection/write-cv.gpqr.py _temp/X.csv _temp/y.csv _temp/%.prior_mean.pt
	python3 $^ --target $* --model CenterGapMTGPQR_$* --num-folds=10 --quantiles $(QUANTILES) --num-lower-quantiles $(NUM_LOWER_QUANTILES) --num-latents $(NUM_LATENTS) --n-epochs $(HEAVYEDGE_N_EPOCHS) -o $@


# Model

_temp/%.prior_mean.pt: scripts/train/prior_mean.py _temp/X.csv _temp/y.csv
	python3 $(wordlist 1,3,$^) --target $* --model PriorMean_$* --num-epochs $(HEAVYEDGE_N_EPOCHS) -o $@

_temp/best-config.%.quantiles.epoch: scripts/train/write-best.py _temp/cv.CenterGapMTGPQR_%.csv
	python3 $^ --target epoch -o $@

model/%.gpqr.pt: scripts/train/gpqr.py _temp/X.csv _temp/y.csv _temp/%.prior_mean.pt _temp/best-config.%.quantiles.epoch
	python3 $(wordlist 1,4,$^) --target $* --model CenterGapMTGPQR_$* --quantiles $(QUANTILES) --num-lower-quantiles $(NUM_LOWER_QUANTILES) --num-latents $(NUM_LATENTS) --num-epochs $(shell cat $(lastword $^)) -o $@

model/%.py: scripts/model/%.py
	cp $< $@

model/requirements.txt: requirements.txt
	grep -E '^(numpy|torch|gpytorch-qr|gpytorch)([>=<!~,; \t]|$$)' $< > $@


# Prediction

_temp/%.prior_mean.Xpred_1D.npy: model/predict-mean.py _temp/Xpred_1D.npy $(MODEL_FILES)
	python3 $(wordlist 1,2,$^) --target $* -o $@

_temp/%.prior_mean.Xpred_2D.npy: model/predict-mean.py _temp/Xpred_2D.npy $(MODEL_FILES)
	python3 $(wordlist 1,2,$^) --target $* -o $@

_temp/%.quantiles.X.npy: model/predict-quantiles.py _temp/X.npy $(MODEL_FILES)
	python3 $(wordlist 1,2,$^) --target $* --method delta -o $@

_temp/%.quantiles.Xpred_1D.npy: model/predict-quantiles.py _temp/Xpred_1D.npy $(MODEL_FILES)
	python3 $(wordlist 1,2,$^) --target $* --method delta -o $@

_temp/%.quantiles.Xpred_2D.npy: model/predict-quantiles.py _temp/Xpred_2D.npy $(MODEL_FILES)
	python3 $(wordlist 1,2,$^) --target $* --method delta -o $@


# Window prediction

_temp/delaunay.Xpred_2D.npy: scripts/data/compute-Delaunay.py _temp/X.csv _temp/Xpred_2D.csv
	python3 $^ --grid Gap_to_thickness_ratio Capillary_number -o $@

_temp/%.pit.Xpred_2D.npy: scripts/joint/write-pit.py _temp/y.csv _temp/%.quantiles.X.npy
	python3 $^ --target $* --quantiles $(QUANTILES) -o $@

_temp/H.marginal.Xpred_2D.npy: scripts/joint/write-marginal.py _temp/H.quantiles.Xpred_2D.npy
	python3 $^ --quantiles $(QUANTILES) --threshold 1.1 -o $@

_temp/phi.marginal.Xpred_2D.npy: scripts/joint/write-marginal.py _temp/phi.quantiles.Xpred_2D.npy
	python3 $^ --quantiles $(QUANTILES) --threshold 1.0 -o $@

_temp/%.pit_marginal.Xpred_2D.npz: _temp/%.pit.Xpred_2D.npy _temp/%.marginal.Xpred_2D.npy
	python3 -c "import numpy as np; pit, marginal = map(np.load, '$^'.split(' ')); np.savez('$@', pit=pit, marginal=marginal)"

_temp/joint_probability.Xpred_2D.npy: scripts/joint/write-joint.py _temp/Xpred_2D.csv _temp/H.pit_marginal.Xpred_2D.npz _temp/phi.pit_marginal.Xpred_2D.npz
	python3 $^ -o $@
