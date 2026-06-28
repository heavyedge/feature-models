NOTEBOOKS := $(wildcard notebooks/*)

.ONESHELL:
.PHONY: all notebooks clean test FORCE

all: \
model/GPR.H.pt \
model/GPR.b.pt \
model/GPR.phi.pt \
model/GPQR.H.pt \
model/GPQR.phi.pt \
model/__init__.py \
model/prior.py \
model/model.py \
model/load.py

notebooks: $(NOTEBOOKS)

clean:
	rm -rf _temp _artifacts model/*.pt model/*.py

test: _temp/test-X.npy
	python3 -c "import torch; import numpy as np; from model.load import gpr_H; gpr_H()[0](torch.tensor(np.load('$<')))"
	python3 -c "import torch; import numpy as np; from model.load import gpr_b; gpr_b()[0](torch.tensor(np.load('$<')))"
	python3 -c "import torch; import numpy as np; from model.load import gpr_phi; gpr_phi()[0](torch.tensor(np.load('$<')))"
	python3 -c "import torch; import numpy as np; from model.load import gpqr_H; gpqr_H()[0](torch.tensor(np.load('$<')))"
	python3 -c "import torch; import numpy as np; from model.load import gpqr_phi; gpqr_phi()[0](torch.tensor(np.load('$<')))"

_temp/test-X.npy:
	mkdir -p $(@D)
	python3 -c "import numpy as np; np.save('$@', np.array([[2.0, 0.2, 0.3]]))"

# Notebooks

notebooks/Crossing.%.ipynb: _temp/X.csv _temp/y.csv FORCE
	jupyter nbconvert --to notebook --execute --inplace $@

notebooks/Extrapolation.%.ipynb: _temp/X.csv _temp/y.csv FORCE
	jupyter nbconvert --to notebook --execute --inplace $@

notebooks/CV.%.ipynb: _temp/X.csv _temp/y.csv FORCE
	jupyter nbconvert --to notebook --execute --inplace $@

notebooks/GPR.ipynb: _temp/X.csv _temp/y.csv model/GPR.H.pt model/GPR.b.pt model/GPR.phi.pt FORCE
	jupyter nbconvert --to notebook --execute --inplace $@

notebooks/GPQR.%.ipynb: _temp/X.csv _temp/y.csv model/GPQR.%.pt FORCE
	jupyter nbconvert --to notebook --execute --inplace $@

notebooks/QW.GPQR.ipynb: _temp/X.csv model/GPQR.H.pt model/GPQR.phi.pt FORCE
	jupyter nbconvert --to notebook --execute --inplace $@

FORCE:  # dummy target to force execution of dependent targets

# Data

_temp/Dataset.csv: scripts/filter-dataset.py _data/Dataset.csv
	mkdir -p $(@D)
	python3 $^ -o $@

_temp/X.csv: _temp/Dataset.csv
	python3 -c "import pandas as pd; import numpy as np; df = pd.read_csv('$<')[['Slurry', 'Gap_to_thickness_ratio', 'Capillary_number', 'Contact_angle']]; df['Cos_theta'] = np.cos(np.radians(df['Contact_angle'])); df.drop('Contact_angle', axis=1).to_csv('$@', index=False)"

_temp/y.csv: _temp/Dataset.csv
	python3 -c "import pandas as pd; pd.read_csv('$<')[['H', 'b', 'phi']].to_csv('$@', index=False)"

model/GPR.%.pt: scripts/train-gpr.py _temp/X.csv _temp/y.csv
	python3 $^ --target $* -o $@

_temp/H.CgLmcMtgpqr.pt: scripts/train-qr.py _temp/X.csv _temp/y.csv
	python3 $^ --target H --model CgLmcMtgpqr --num-epochs 2706 -o $@

_temp/phi.CgIndependentMtgpqr.pt: scripts/train-qr.py _temp/X.csv _temp/y.csv
	python3 $^ --target phi --model CgIndependentMtgpqr --num-epochs 9543 -o $@

model/GPQR.H.pt: _temp/H.CgLmcMtgpqr.pt
	cp $< $@

model/GPQR.phi.pt: _temp/phi.CgIndependentMtgpqr.pt
	cp $< $@

_temp/window.H.npy: _temp/y.csv
	python3 -c "import pandas as pd; import numpy as np; np.save('$@', pd.read_csv('$<')['H'].apply(lambda x: x <= 1.1).to_numpy())"

_temp/window.phi.npy: _temp/y.csv
	python3 -c "import pandas as pd; import numpy as np; np.save('$@', pd.read_csv('$<')['phi'].apply(lambda x: x <= 1.0).to_numpy())"

_temp/window.npy: _temp/window.H.npy _temp/window.phi.npy
	python3 -c "import numpy as np; qw = np.all([np.load(f) for f in '$^'.split(' ')], axis=0).flatten(); np.save('$@', qw)"

model/%.py: scripts/%.py
	cp $< $@
