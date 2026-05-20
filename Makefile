NOTEBOOKS := $(wildcard notebooks/*)

.ONESHELL:
.PHONY: all notebooks clean FORCE

all: \
model/H.pt \
model/phi.pt

notebooks: $(NOTEBOOKS)

clean:
	rm -rf _temp _artifacts model/*.pt

# Notebooks

notebooks/CrossValidation.%.ipynb: _temp/X.csv _temp/y.csv FORCE
	jupyter nbconvert --to notebook --execute --inplace $@

notebooks/Model.%.ipynb: _temp/X.csv _temp/y.csv model/%.pt FORCE
	jupyter nbconvert --to notebook --execute --inplace $@

FORCE:  # dummy target to force execution of dependent targets

# Data

_temp/Dataset.csv: scripts/filter-dataset.py _data/Dataset.csv
	@mkdir -p $(@D)
	python3 $^ -o $@

_temp/X.csv: _temp/Dataset.csv
	python3 -c "import pandas as pd; import numpy as np; df = pd.read_csv('$<')[['Gap_to_thickness_ratio', 'Capillary_number', 'Contact_angle']]; df['Cos_theta'] = np.cos(np.radians(df['Contact_angle'])); df.drop('Contact_angle', axis=1).to_csv('$@', index=False)"

_temp/y.csv: _temp/Dataset.csv
	python3 -c "import pandas as pd; pd.read_csv('$<')[['H', 'phi']].to_csv('$@', index=False)"

_temp/H.MTGPQR.pt: scripts/train.py _temp/X.csv _temp/y.csv
	python3 $^ --target H --model MTGPQR --num-epochs 3127 -o $@

_temp/phi.MTGPQR.pt: scripts/train.py _temp/X.csv _temp/y.csv
	python3 $^ --target phi --model MTGPQR --num-epochs 5764 -o $@

model/H.pt: _temp/H.MTGPQR.pt
	cp $< $@

model/phi.pt: _temp/phi.MTGPQR.pt
	cp $< $@
