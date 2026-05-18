.ONESHELL:
.PHONY: all clean

all:

clean:
	rm -rf _temp _artifacts

_temp/Dataset.csv: scripts/filter-dataset.py _data/Dataset.csv
	@mkdir -p $(@D)
	python3 $^ -o $@

_temp/X.csv: _temp/Dataset.csv
	@mkdir -p $(@D)
	python3 -c "import pandas as pd; import numpy as np; df = pd.read_csv('$<')[['Gap_to_thickness_ratio', 'Capillary_number', 'Contact_angle']]; df['Cos_theta'] = np.cos(np.radians(df['Contact_angle'])); df.drop('Contact_angle', axis=1).to_csv('$@', index=False)"

_temp/y.csv: _temp/Dataset.csv
	@mkdir -p $(@D)
	python3 -c "import pandas as pd; pd.read_csv('$<')[['H', 'phi']].to_csv('$@', index=False)"

_temp/H.%.CV.pkl: scripts/cv.py _temp/X.csv _temp/y.csv
	python3 $^ --target H --model $* --n-epochs 5000 -o $@
