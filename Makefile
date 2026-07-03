NOTEBOOKS := $(wildcard notebooks/*)
NUM_LATENTS = 3
HEAVYEDGE_N_EPOCHS ?= 10000

.SECONDARY:
.ONESHELL:
.PHONY: models notebooks test all clean FORCE

models: \
model/H.gpqr.pt \
model/phi.gpqr.pt \
model/prior.py \
model/scale.py \
model/gpqr.py \
model/load.py \
model/requirements.txt

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

notebooks/Quantiles.ipynb: _temp/X.csv _temp/y.csv models FORCE
	jupyter nbconvert --to notebook --execute --inplace $@

notebooks/Window.ipynb: _temp/X.csv _temp/X-pred.csv _temp/joint_probability.X-pred.npz _temp/X-delaunay.npy FORCE
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

_temp/X-pred.csv: scripts/data/write-Xpred.py _temp/X.csv
	python3 $^ -o $@

_temp/X.npy: _temp/X.csv
	python3 -c "import pandas as pd; import numpy as np; np.save('$@', pd.read_csv('$<').drop(columns=['Slurry']).to_numpy())"

_temp/X-pred.npy: _temp/X-pred.csv
	python3 -c "import pandas as pd; import numpy as np; df = pd.read_csv('$<', index_col=[0,1,2]); shape = [df.index.get_level_values(i).nunique() for i in range(df.index.nlevels)]; np.save('$@', df.to_numpy().reshape(*shape, -1))"

_temp/X-test1.csv: scripts/data/write-Xtest.py _temp/X.csv
	python3 $^ --start=0 --stop=1 --num=10 -o $@

_temp/X-test2.csv: scripts/data/write-Xtest.py _temp/X.csv
	python3 $^ --start=-2 --stop=2 --num=10 -o $@

# Model selection

_temp/crossing.DirectMTGPQR_%.csv: scripts/model_selection/write-crossing.py _temp/X.csv _temp/y.csv _temp/%.prior_mean.pt _temp/X-test1.csv _temp/X-test2.csv
	python3 $^ --target $* --model DirectMTGPQR_$* --quantiles 0.05 0.25 0.5 0.75 0.95 --num-latents $(NUM_LATENTS) --n-epochs $(HEAVYEDGE_N_EPOCHS) -o $@

_temp/extrapolation.CenterGapMTGPQR_%.csv: scripts/model_selection/write-extrapolation.py _temp/X.csv _temp/y.csv _temp/%.prior_mean.pt
	python3 $^ --target $* --model CenterGapMTGPQR_$* --split-ratio=0.8 --quantiles 0.05 0.25 0.5 0.75 0.95 --num-lower-quantiles 2 --num-latents $(NUM_LATENTS) --n-epochs $(HEAVYEDGE_N_EPOCHS) -o $@

_temp/extrapolation.CenterGapMTGPQR_%_ConstantMean.csv: scripts/model_selection/write-extrapolation.py _temp/X.csv _temp/y.csv
	python3 $^ --target $* --model CenterGapMTGPQR_$* --split-ratio=0.8 --quantiles 0.05 0.25 0.5 0.75 0.95 --num-lower-quantiles 2 --num-latents $(NUM_LATENTS) --n-epochs $(HEAVYEDGE_N_EPOCHS) -o $@

_temp/cv.GPR_%.csv: scripts/model_selection/write-cv.gpr.py _temp/X.csv _temp/y.csv
	python3 $^ --target $* --model GPR_$* --num-folds=10 --quantiles 0.05 0.25 0.5 0.75 0.95 --n-epochs $(HEAVYEDGE_N_EPOCHS) -o $@

_temp/cv.CenterGapMTGPQR_%.csv: scripts/model_selection/write-cv.gpqr.py _temp/X.csv _temp/y.csv _temp/%.prior_mean.pt
	python3 $^ --target $* --model CenterGapMTGPQR_$* --num-folds=10 --quantiles 0.05 0.25 0.5 0.75 0.95 --num-lower-quantiles 2 --num-latents $(NUM_LATENTS) --n-epochs $(HEAVYEDGE_N_EPOCHS) -o $@


# Model

_temp/%.prior_mean.pt: scripts/train/prior_mean.py _temp/X.csv _temp/y.csv
	python3 $(wordlist 1,3,$^) --target $* --model PriorMean_$* --num-epochs $(HEAVYEDGE_N_EPOCHS) -o $@

_temp/best-config.%.quantiles.epoch: scripts/train/write-best.py _temp/cv.CenterGapMTGPQR_%.csv
	python3 $^ --target epoch -o $@

model/%.gpqr.pt: scripts/train/gpqr.py _temp/X.csv _temp/y.csv _temp/%.prior_mean.pt _temp/best-config.%.quantiles.epoch
	python3 $(wordlist 1,4,$^) --target $* --model CenterGapMTGPQR_$* --quantiles 0.05 0.25 0.5 0.75 0.95 --num-lower-quantiles 2 --num-latents $(NUM_LATENTS) --num-epochs $(shell cat $(lastword $^)) -o $@

model/%.py: scripts/model/%.py
	cp $< $@

model/requirements.txt: requirements.txt
	grep -E '^(numpy|torch|gpytorch-qr|gpytorch)([>=<!~,; \t]|$$)' $< > $@

# Window prediction

_temp/%.quantiles.X.npz: scripts/predict/gpqr.py _temp/X.npy model/%.gpqr.pt
	python3 $(wordlist 1,2,$^) $(abspath $(lastword $^)) --target $* -o $@

_temp/%.quantiles.X-pred.npz: scripts/predict/gpqr.py _temp/X-pred.npy model/%.gpqr.pt
	python3 $(wordlist 1,2,$^) $(abspath $(lastword $^)) --target $* -o $@

_temp/H.pit.X-pred.npz: scripts/joint/write-pit.py _temp/y.csv _temp/H.quantiles.X.npz _temp/H.quantiles.X-pred.npz
	python3 $^ --target H --threshold 1.1 -o $@

_temp/phi.pit.X-pred.npz: scripts/joint/write-pit.py _temp/y.csv _temp/phi.quantiles.X.npz _temp/phi.quantiles.X-pred.npz
	python3 $^ --target phi --threshold 1.0 -o $@

_temp/joint_probability.X-pred.npz: scripts/joint/write-joint.py _temp/X.csv _temp/X-pred.csv _temp/H.pit.X-pred.npz _temp/phi.pit.X-pred.npz
	python3 $^ -o $@

_temp/X-delaunay.npy: scripts/data/compute-Delaunay.py _temp/X.csv _temp/X-pred.csv
	python3 $^ -o $@
