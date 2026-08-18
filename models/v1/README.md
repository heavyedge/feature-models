---
license: mit
language:
- en
datasets:
- heavyedge/shape-features
new_version: heavyedge/feature-model-v1
---
# HeavyEdge Shape Feature Model v1

Setup:

```
pip install -r requirements.txt
```

Usage:

```
python -m feature_models.predict-prior_mean <args>
python -m feature_models.predict-gpr <args>
python -m feature_models.predict-gpqr <args>
```
