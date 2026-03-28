.ONESHELL:
.PHONY: all clean

all: model/H-model.pth

clean:
	rm -rf _temp model/H-model.pth

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
	@mkdir -p $(@D)
	python3 $^ --target H -o $@
