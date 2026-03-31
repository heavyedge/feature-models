.ONESHELL:
.PHONY: all clean

all: \
model/H-model.pth model/model.py \
_artifacts/H-model.png _artifacts/H-probability.png

clean:
	rm -rf _temp _artifacts model/H-model.pth model/model.py

_temp/Dataset.csv: scripts/filter-dataset.py _data/Dataset.csv
	@mkdir -p $(@D)
	python3 $^ -o $@

_temp/X.csv: _temp/Dataset.csv
	@mkdir -p $(@D)
	python3 -c "import pandas as pd; pd.read_csv('$<')[['Gap_to_thickness_ratio', 'Capillary_number', 'Surface_tension']].to_csv('$@', index=False)"

_temp/y.csv: _temp/Dataset.csv
	@mkdir -p $(@D)
	python3 -c "import pandas as pd; pd.read_csv('$<')[['H']].to_csv('$@', index=False)"

model/model.py: scripts/model.py
	cp $< $@

model/H-model.pth: scripts/train-qr.py _temp/X.csv _temp/y.csv
	python3 $^ --target H -o $@

_artifacts/H-model.png: scripts/plot-model.py _temp/X.csv model/H-model.pth _temp/y.csv
	@mkdir -p $(@D)
	python3 $^ --target H -o $@

_artifacts/H-probability.png: scripts/plot-probability.py _temp/X.csv model/H-model.pth
	@mkdir -p $(@D)
	python3 $^ --target H -o $@
