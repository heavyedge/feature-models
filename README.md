# HeavyEdge-Features quantile regression model

Repository to train and distribute quantile regression model of edge shape features.

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

### Re-building examples

Configure the local git filter (run once after cloning):

```
git config filter.nbstripout.clean "nbstripout --keep-output --keep-metadata-keys 'metadata.language_info'"
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true
```

Then build the examples:

```
make notebooks
```
