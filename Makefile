.ONESHELL:
.PHONY: all clean

all: model/H-model.pth, model/model.py

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
	python3 -c "import pandas as pd; pd.read_csv('$<')[['phi', 'H', 'b']].to_csv('$@', index=False)"

model/H-model.pth: scripts/train-qr.py _temp/X.csv _temp/y.csv
	python3 $^ --target H -o $@

_artifacts/H-model.png: scripts/plot-qr.py _temp/X.csv model/H-model.pth
	@mkdir -p $(@D)
	python3 $^ --target H -o $@

model/model.py: scripts/model.py
	cp $< $@
