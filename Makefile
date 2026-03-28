.ONESHELL:
.PHONY: all clean

all:

clean:
	rm -rf _temp

_temp/Dataset.csv: scripts/filter-dataset.py _data/Dataset.csv
	@mkdir -p $(@D)
	python3 $^ -o $@
