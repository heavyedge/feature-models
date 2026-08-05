.PHONY: models-v1 examples-v1

PYTHON ?= python3
GPU_RUN ?= $(PYTHON) scripts/gpu-run.py --
GPU_PYTHON ?= $(GPU_RUN) $(PYTHON)
GPU_JUPYTER ?= $(GPU_RUN) jupyter

MODEL_FILES_v1 := \
models/v1/feature_models/H.prior_mean.pt \
models/v1/feature_models/phi.prior_mean.pt

models-v1: $(MODEL_FILES_v1)

examples-v1: $(wildcard examples/v1/*)

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

# Models

## Prior mean

models/v1/feature_models/%.prior_mean.pt: scripts/v1/train/prior_mean.py _temp/v1/Xtrain.csv _temp/v1/ytrain.csv _temp/v1/Xval.csv _temp/v1/yval.csv
	mkdir -p $(@D)
	PYTHONPATH=scripts/v0 $(GPU_PYTHON) $^ --target $* --model PriorMean_$* --num-epochs $(N_EPOCHS) -o $@

# Examples

examples/v1/Models.ipynb: _temp/v1/X.csv _temp/v1/y.csv _temp/v0/Xpred_1D.csv .FORCE
	$(GPU_JUPYTER) nbconvert --to notebook --execute --inplace $@
