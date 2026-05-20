.ONESHELL:
.PHONY: all clean

all: \
_artifacts/H.prior.initial.png \
_artifacts/H.prior.trained.png \
_artifacts/H.MTGPQR.quantiles.png \
_artifacts/phi.MTGPQR.quantiles.png \
model/H.pt \
model/phi.pt

clean:
	rm -rf _temp _artifacts model/*.pt

# Figures

## CV

## Prior

_artifacts/H.prior.initial.png: scripts/plot-prior.initial.py _temp/X.csv _temp/y.csv
	mkdir -p $(@D)
	python3 $^ --target H -o $@

_artifacts/H.prior.trained.png: scripts/plot-prior.trained.py _temp/X.csv _temp/y.csv _temp/H.MTGPQR.pt
	mkdir -p $(@D)
	python3 $^ --target H --model MTGPQR -o $@

## Quantiles

_artifacts/H.%.quantiles.png: scripts/plot-quantiles.py _temp/X.csv _temp/y.csv _temp/H.%.pt
	mkdir -p $(@D)
	python3 $^ --target H --model $* -o $@

_artifacts/phi.%.quantiles.png: scripts/plot-quantiles.py _temp/X.csv _temp/y.csv _temp/phi.%.pt
	mkdir -p $(@D)
	python3 $^ --target phi --model $* -o $@

# Data

_temp/Dataset.csv: scripts/filter-dataset.py _data/Dataset.csv
	@mkdir -p $(@D)
	python3 $^ -o $@

_temp/X.csv: _temp/Dataset.csv
	python3 -c "import pandas as pd; import numpy as np; df = pd.read_csv('$<')[['Gap_to_thickness_ratio', 'Capillary_number', 'Contact_angle']]; df['Cos_theta'] = np.cos(np.radians(df['Contact_angle'])); df.drop('Contact_angle', axis=1).to_csv('$@', index=False)"

_temp/y.csv: _temp/Dataset.csv
	python3 -c "import pandas as pd; pd.read_csv('$<')[['H', 'phi']].to_csv('$@', index=False)"

_temp/H.MTGPQR.CV.csv: scripts/cv.py _temp/X.csv _temp/y.csv
	python3 $^ --target H --model MTGPQR --n-epochs 5000 -o $@

_temp/phi.MTGPQR.CV.csv: scripts/cv.py _temp/X.csv _temp/y.csv
	python3 $^ --target phi --model MTGPQR --n-epochs 5000 -o $@

_temp/H.MTGPQR.pt: scripts/train.py _temp/X.csv _temp/y.csv
	python3 $^ --target H --model MTGPQR -o $@

_temp/phi.MTGPQR.pt: scripts/train.py _temp/X.csv _temp/y.csv
	python3 $^ --target phi --model MTGPQR -o $@

model/H.pt: _temp/H.MTGPQR.pt
	cp $< $@

model/phi.pt: _temp/phi.MTGPQR.pt
	cp $< $@
