# Edge shape feature model
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/jeesoo9595/heavyedge-features-v1)
[![GitHub repository](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/heavyedge/feature-models)

Repository to train and distribute Gaussian process quantile regression (GPQR) models of heavy edge shape features.

Provides:
- GPQR models of H and phi.
- Cross validation and other validation examples.
- Yield estimation and probabilistic quality window examples.

Trained models are distributed in [Huggingface repository](https://huggingface.co/jeesoo9595/heavyedge-features-v1).

## Download feature data

```
curl -LsSf https://hf.co/cli/install.sh | bash
hf auth login --token [Huggingface token]
./setup.sh
```

## Install prereqisites

```
uv pip install --system -r requirements.txt
```

## Train & plot

```
make models
make notebooks
```

You would want to run these commands in the CUDA environment.

### Testing built models

```
make test
```

## Documentation

> https://heavyedge.github.io/feature-models/

To build the documents yourself, run the following commands:

```
make notebooks
cd doc
make html
```

Document will be generated in `build/html` directory. Open `index.html` to see the central page.

## Developing

### Configuring git

Configure the local git filter (run once after cloning):

```
git config filter.nbstripout.clean "nbstripout"
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true
```

### Building models using Docker

Store Huggingface token in `HF_TOKEN` environment variable and run the following command:

```
docker build --secret id=hf_token,env=HF_TOKEN --target models --output type=local,dest=./model .
```

### Building notebooks using Docker

Store Huggingface token in `HF_TOKEN` environment variable and run the following command:

```
docker build --secret id=hf_token,env=HF_TOKEN --target notebooks --output type=local,dest=./notebooks .
```

## Versioning policy

This repository follows semantic versioning with [Python version specifiers](https://packaging.python.org/en/latest/specifications/version-specifiers/):

```
N.N.N[.postN]
```

**Major version**

- Updated if model API is changed.
- Each major version has dedicated repository, e.g., `heavyedge-features-v1`, `heavyedge-features-v2`, and so on.

**Minor version**

- Updated if model is re-trained without API change.

**Patch version**

- Bug fix.
- Model metadata change.

**Post release**

- Changes that do not affect models, e.g., documentation update.
- Post releases are not pushed to Huggingface repository.
