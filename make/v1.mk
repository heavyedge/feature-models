.PHONY:

# Data

_temp/v1/dimless.csv: $(wildcard _data/v1/dimless/all_profiles/dataset*.csv)
	mkdir -p $(@D)
	python3 -c '
	from pathlib import Path
	import pandas as pd

	pvs = sorted([Path(f) for f in "$^".split(" ")])
	df = pd.concat(
		[pd.read_csv(f, dtype=str).assign(name=lambda df: f"{f.stem}/" + df["name"]) for f in pvs],
		ignore_index=True,
	)
	df.to_csv("$@", index=False)
	'

_temp/v1/shape_features.csv: $(wildcard _data/v1/shape_features/profiles/dataset*.csv)
	mkdir -p $(@D)
	python3 -c '
	from pathlib import Path
	import pandas as pd

	pvs = sorted([Path(f) for f in "$^".split(" ")])
	df = pd.concat(
		[pd.read_csv(f, dtype=str).assign(name=lambda df: f"{f.stem}/" + df["name"]) for f in pvs],
		ignore_index=True,
	)
	df.to_csv("$@", index=False)
	'

_temp/v1/X.csv: scripts/v1/data/write-X.py _temp/v1/dimless.csv
	mkdir -p $(@D)
	python3 $^ --split-ratio 0.8 0.1 0.1 --random-state=42 -o $@

_temp/v1/y.csv: scripts/v1/data/write-y.py _temp/v1/X.csv _temp/v1/shape_features.csv
	mkdir -p $(@D)
	python3 $^ -o $@

define SPLIT_v1
_temp/v1/X$(1).csv: _temp/v1/X.csv
	python3 -c "import pandas as pd; df = pd.read_csv('$$<'); mask = df['split'] == '$(1)'; df.loc[mask, ['gap_to_thickness_ratio', 'capillary_number', 'cosine_of_contact_angle']].to_csv('$$@', index=False)"
_temp/v1/y$(1).csv: _temp/v1/y.csv
	python3 -c "import pandas as pd; df = pd.read_csv('$$<'); mask = df['split'] == '$(1)'; df.loc[mask, ['H', 'phi']].to_csv('$$@', index=False)"
endef
$(foreach split,train val test,$(eval $(call SPLIT_v1,$(split))))
