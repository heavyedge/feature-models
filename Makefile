.ONESHELL:
.PHONY: all clean

all: \
_artifacts/H.CV.epoch.png \
_artifacts/H.CV.min.png \
_artifacts/phi.CV.epoch.png \
_artifacts/phi.CV.min.png \
_artifacts/H.prior.initial.png \
_artifacts/phi.prior.initial.png \
_artifacts/H.CgLmcMtgpqr.quantiles.png \
_artifacts/phi.CgLmcMtgpqr.quantiles.png \
model/H.pt \
model/phi.pt

clean:
	rm -rf _temp _artifacts model/*.pt

_temp/Dataset.csv: scripts/filter-dataset.py _data/Dataset.csv
	@mkdir -p $(@D)
	python3 $^ -o $@

_temp/X.csv: _temp/Dataset.csv
	python3 -c "import pandas as pd; import numpy as np; df = pd.read_csv('$<')[['Gap_to_thickness_ratio', 'Capillary_number', 'Contact_angle']]; df['Cos_theta'] = np.cos(np.radians(df['Contact_angle'])); df.drop('Contact_angle', axis=1).to_csv('$@', index=False)"

_temp/y.csv: _temp/Dataset.csv
	python3 -c "import pandas as pd; pd.read_csv('$<')[['H', 'phi']].to_csv('$@', index=False)"

_temp/H.%.CV.csv: scripts/cv.py _temp/X.csv _temp/y.csv
	python3 $^ --target H --model $* --n-epochs 5000 -o $@

_temp/phi.%.CV.csv: scripts/cv.py _temp/X.csv _temp/y.csv
	python3 $^ --target phi --model $* --n-epochs 5000 -o $@

_artifacts/H.CV.epoch.png: scripts/plot-cv.epoch.py _temp/H.CgLmcMtgpqr.CV.csv _temp/H.DirectLmcMtgpqr.CV.csv _temp/H.CgIndependentMtgpqr.CV.csv _temp/H.DirectIndependentMtgpqr.CV.csv
	mkdir -p $(@D)
	python3 $^ -o $@

_artifacts/H.CV.min.png: scripts/plot-cv.min.py _temp/H.CgLmcMtgpqr.CV.csv _temp/H.DirectLmcMtgpqr.CV.csv _temp/H.CgIndependentMtgpqr.CV.csv _temp/H.DirectIndependentMtgpqr.CV.csv
	mkdir -p $(@D)
	python3 $^ --ymin 5.5e-3 -o $@

_artifacts/phi.CV.epoch.png: scripts/plot-cv.epoch.py _temp/phi.CgLmcMtgpqr.CV.csv _temp/phi.DirectLmcMtgpqr.CV.csv _temp/phi.CgIndependentMtgpqr.CV.csv _temp/phi.DirectIndependentMtgpqr.CV.csv
	mkdir -p $(@D)
	python3 $^ -o $@

_artifacts/phi.CV.min.png: scripts/plot-cv.min.py _temp/phi.CgLmcMtgpqr.CV.csv _temp/phi.DirectLmcMtgpqr.CV.csv _temp/phi.CgIndependentMtgpqr.CV.csv _temp/phi.DirectIndependentMtgpqr.CV.csv
	mkdir -p $(@D)
	python3 $^ --ymin 5.5e-3 -o $@

_temp/H.%.pt: scripts/train.py _temp/X.csv _temp/y.csv _temp/H.%.CV.csv
	python3 $^ --target H --model $* -o $@

_temp/phi.%.pt: scripts/train.py _temp/X.csv _temp/y.csv _temp/phi.%.CV.csv
	python3 $^ --target phi --model $* -o $@

_artifacts/H.prior.initial.png: scripts/plot-prior.initial.py _temp/X.csv _temp/y.csv
	mkdir -p $(@D)
	python3 $^ --target H -o $@

_artifacts/phi.prior.initial.png: scripts/plot-prior.initial.py _temp/X.csv _temp/y.csv
	mkdir -p $(@D)
	python3 $^ --target phi -o $@

_artifacts/H.%.quantiles.png: scripts/plot-quantiles.py _temp/X.csv _temp/y.csv _temp/H.%.pt
	mkdir -p $(@D)
	python3 $^ --target H --model $* -o $@

_artifacts/phi.%.quantiles.png: scripts/plot-quantiles.py _temp/X.csv _temp/y.csv _temp/phi.%.pt
	mkdir -p $(@D)
	python3 $^ --target phi --model $* -o $@

model/H.pt: _temp/H.CgLmcMtgpqr.pt
	cp $< $@

model/phi.pt: _temp/phi.CgLmcMtgpqr.pt
	cp $< $@
