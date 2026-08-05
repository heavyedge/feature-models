---
license: mit
language:
- en
datasets:
- heavyedge/shape-features
---
# HeavyEdge Shape Feature Model v1

Setup:

```
pip install -r requirements.txt
```

Usage:

Run `predict-prior_mean.py` script to evaluate prior mean function.
Run `predict-mean.py` script to evaluate posterior distribution of mean function.

Help:

```
python feature_models/predict-prior_mean.py -h
python feature_models/predict-mean.py -h
```