# Edge shape feature model

Repository to train and distribute models related to heavy edge shape features.

- Gaussian process quantile regression models.
- Quality window models (deterministic and probabilistic).

## Download feature data

```
curl -LsSf https://hf.co/cli/install.sh | bash
hf download jeesoo9595/heavyedge-features-v1 --repo-type dataset --revision v1.1.0 --local-dir _data
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
