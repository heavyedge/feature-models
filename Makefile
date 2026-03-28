.ONESHELL:
.PHONY: all clean

all:

clean:
	rm -rf _temp

_temp/Dataset.csv: scripts/filter-dataset.py _data/Dataset.csv
	@mkdir -p $(@D)
	python3 $^ -o $@

_temp/X.csv: _temp/Dataset.csv
	@mkdir -p $(@D)
	python3 -c "import pandas as pd; pd.read_csv('$<')[['Gap_to_thickness_ratio', 'Capillary_number', 'Surface_tension']].to_csv('$@', index=False)"

_temp/y.csv: _temp/Dataset.csv
	@mkdir -p $(@D)
	python3 -c "import pandas as pd; pd.read_csv('$<')[['phi', 'H', 'b']].to_csv('$@', index=False)"

_temp/X-scaler.pkl: scripts/train-scaler.py _temp/X.csv
	@mkdir -p $(@D)
	python3 $^ -o $@
