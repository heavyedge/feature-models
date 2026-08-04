---
license: mit
language:
- en
datasets:
- heavyedge/shape-features
---
# HeavyEdge Shape Feature Model v0

Setup:

```
pip install -r requirements.txt
```

Usage:

Run `predict-prior_mean.py` script to evaluate prior mean function.
Run `predict-mean.py` script to evaluate posterior distribution of mean function.
Run `predict-quantiles.py` script to evaluate quantile funtions.

Help:

```
python predict-prior_mean.py -h
python predict-mean.py -h
python predict-quantiles.py -h
```