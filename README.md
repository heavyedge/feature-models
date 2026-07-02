# Edge shape feature model
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/jeesoo9595/heavyedge-features-v1)
[![GitHub repository](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/heavyedge/feature-models)

Repository to train and distribute models related to heavy edge shape features.

- Gaussian process quantile regression models.
- Quality window models (deterministic and probabilistic).

You would want to build this repository in the CUDA environment.

## Download feature data

```
curl -LsSf https://hf.co/cli/install.sh | bash
hf auth login --token [Huggingface token]
./setup.sh
```

## Install prereqisites

```
pip install -r requirements.txt
```

## Train & plot

```
make models
make notebooks
```

### Testing built models

```
make test
```

## Developing

### Configuring git

Configure the local git filter (run once after cloning):

```
git config filter.nbstripout.clean "nbstripout --keep-output --keep-metadata-keys 'metadata.language_info'"
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

The HeavyEdge-Features model follows semantic versioning.

**Major version**

- Updated if model API is changed.
- Each major version has dedicated repository, e.g., `heavyedge-features-v1`, `heavyedge-features-v2`, and so on.

**Minor version**

- Updated if model is re-trained without API change.

**Patch version**

- Bug fix.
- Metadata change.
