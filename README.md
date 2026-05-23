# Edge shape feature model
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/jeesoo9595/heavyedge-features-v1)
[![GitHub repository](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/heavyedge/feature-models)

Repository to train and distribute models related to heavy edge shape features.

- Gaussian process quantile regression models.
- Quality window models (deterministic and probabilistic).

## Download feature data

```
curl -LsSf https://hf.co/cli/install.sh | bash
hf download jeesoo9595/heavyedge-features-v1 --repo-type dataset --revision v1.3.0 --local-dir _data
```

## Train & plot

```
pip install -r requirements.txt
make
```

You may want to run `make` in the CUDA environment.

## Developing

### Re-building notebooks

Configure the local git filter (run once after cloning):

```
git config filter.nbstripout.clean "nbstripout --keep-output --keep-metadata-keys 'metadata.language_info'"
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true
```

Then build the notebooks:

```
make notebooks
```

### Testing built models

```
make test
```
