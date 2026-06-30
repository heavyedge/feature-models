NOTEBOOKS := $(wildcard notebooks/*)
HEAVYEDGE_N_EPOCHS ?= 10000

.SECONDARY:
.ONESHELL:
.PHONY: all notebooks clean test FORCE

all: \
model/GPR.H.pt \
model/GPR.b.pt \
model/GPR.phi.pt \
model/GPQR.H.pt \
model/GPQR.phi.pt \
model/prior.py \
model/gpr.py \
model/gpqr.py \
model/load.py

notebooks: $(NOTEBOOKS)

clean:
	rm -rf _temp _artifacts model/*.pt model/*.py

test:
	python3 -c "from model.load import load_gpr_H; load_gpr_H()"
	python3 -c "from model.load import load_gpr_b; load_gpr_b()"
	python3 -c "from model.load import load_gpr_phi; load_gpr_phi()"
	python3 -c "from model.load import load_gpqr_H; load_gpqr_H()"
	python3 -c "from model.load import load_gpqr_phi; load_gpqr_phi()"

# Notebooks

notebooks/Crossing.%.ipynb: _temp/X.csv _temp/y.csv _temp/crossing.DirectLmcMtgpqr_%.csv _temp/crossing.DirectIndependentMtgpqr_%.csv FORCE
	jupyter nbconvert --to notebook --execute --inplace $@

notebooks/Extrapolation.%.ipynb: _temp/X.csv _temp/y.csv _temp/extrapolation.GPR_%.csv _temp/extrapolation.GPR_%_ConstantMean.csv FORCE
	jupyter nbconvert --to notebook --execute --inplace $@

notebooks/CV.%.ipynb: _temp/X.csv _temp/y.csv FORCE
	jupyter nbconvert --to notebook --execute --inplace $@

notebooks/GPR.ipynb: _temp/X.csv _temp/y.csv model/GPR.H.pt model/GPR.b.pt model/GPR.phi.pt FORCE
	jupyter nbconvert --to notebook --execute --inplace $@

notebooks/GPQR.%.ipynb: _temp/X.csv _temp/y.csv model/GPQR.%.pt FORCE
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
	python3 -c "import pandas as pd; pd.read_csv('$<')[['H', 'b', 'phi']].to_csv('$@', index=False)"

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

_temp/crossing.DirectLmcMtgpqr_%.csv: scripts/model_selection/write-crossing.py _temp/X.csv _temp/y.csv _temp/X-test1.csv _temp/X-test2.csv
	python3 $^ --model DirectLmcMtgpqr_$* --target $* --quantiles 0.05 0.25 0.5 0.75 0.95 --num-lower-quantiles 2 --num-latents 5 --num-lower-latents 2 --n-epochs $(HEAVYEDGE_N_EPOCHS) -o $@

_temp/crossing.DirectIndependentMtgpqr_%.csv: scripts/model_selection/write-crossing.py _temp/X.csv _temp/y.csv _temp/X-test1.csv _temp/X-test2.csv
	python3 $^ --model DirectIndependentMtgpqr_$* --target $* --quantiles 0.05 0.25 0.5 0.75 0.95 --num-lower-quantiles 2 --num-latents 5 --num-lower-latents 2 --n-epochs $(HEAVYEDGE_N_EPOCHS) -o $@

_temp/extrapolation.GPR_%.csv: scripts/model_selection/write-extrapolation.gpr.py _temp/X.csv _temp/y.csv
	python3 $^ --model GPR_$* --target $* --quantiles 0.05 0.25 0.5 0.75 0.95 --n-epochs $(HEAVYEDGE_N_EPOCHS) --split-ratio=0.8 -o $@

_temp/extrapolation.GPR_%_ConstantMean.csv: scripts/model_selection/write-extrapolation.gpr.py _temp/X.csv _temp/y.csv
	python3 $^ --model GPR_$* --target $* --quantiles 0.05 0.25 0.5 0.75 0.95 --n-epochs $(HEAVYEDGE_N_EPOCHS) --split-ratio=0.8 -o $@

# Model

model/GPR.%.pt: scripts/train/gpr.py _temp/X.csv _temp/y.csv
	python3 $^ --target $* -o $@

_temp/GPQR.H.pt: scripts/train/gpqr.py _temp/X.csv _temp/y.csv
	python3 $^ --target H --model CgLmcMtgpqr --num-epochs 2706 -o $@

_temp/GPQR.phi.pt: scripts/train/gpqr.py _temp/X.csv _temp/y.csv
	python3 $^ --target phi --model CgIndependentMtgpqr --num-epochs 9543 -o $@

model/%.pt: _temp/%.pt
	cp $< $@

model/%.py: scripts/model/%.py
	cp $< $@

# Window prediction

_temp/%.quantiles.X.npz: scripts/predict/gpqr.py _temp/X.npy model/GPQR.%.pt
	python3 $(wordlist 1,2,$^) $(abspath $(lastword $^)) --target $* -o $@

_temp/%.quantiles.X-pred.npz: scripts/predict/gpqr.py _temp/X-pred.npy model/GPQR.%.pt
	python3 $(wordlist 1,2,$^) $(abspath $(lastword $^)) --target $* -o $@

_temp/H.pit.X-pred.npz: scripts/joint/write-pit.py _temp/y.csv _temp/H.quantiles.X.npz _temp/H.quantiles.X-pred.npz
	python3 $^ --target H --threshold 1.1 -o $@

_temp/phi.pit.X-pred.npz: scripts/joint/write-pit.py _temp/y.csv _temp/phi.quantiles.X.npz _temp/phi.quantiles.X-pred.npz
	python3 $^ --target phi --threshold 1.0 -o $@

_temp/joint_probability.X-pred.npz: scripts/joint/write-joint.py _temp/X.csv _temp/X-pred.csv _temp/H.pit.X-pred.npz _temp/phi.pit.X-pred.npz
	python3 $^ -o $@

_temp/X-delaunay.npy: scripts/data/compute-Delaunay.py _temp/X.csv _temp/X-pred.csv
	python3 $^ -o $@
