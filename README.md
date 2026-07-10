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

Documents are built and pushed to Github Pages.
Document for the latest version can be found at:

> https://heavyedge.github.io/feature-models/

other versions can be found by specifying sub-urls.
For example, document for `v1.0.0` can be found at:

> https://heavyedge.github.io/feature-models/v1.0.0

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

## Versioning policy

This repository follows semantic versioning with [Python version specifiers](https://packaging.python.org/en/latest/specifications/version-specifiers/):

```
N.N.N[{a|b|rc}N][.postN][.devN]
```

**Major version**

Major version change indicates that a completely different model is released.
Each major version has dedicated repository, e.g., `heavyedge-features-v1`, `heavyedge-features-v2`, and so on.

- Model inference API MAY change.
- Model architecture and weight MAY change.
- New versions of trained model and inference image MUST be released.

**Minor version**

Minor version change indicates that the same model is re-trained with different data and released.

- Model inference API and architecture MUST NOT change.
- Model weight MAY change.
- New versions of trained model and inference image MUST be released.

**Patch version**

Patch version change indicates metadata change and bug fixes in auxiliary scripts that are distributed with the model.
Trained model itself SHOULD NOT change, unless it is trained with the same data.

- Model inference API and architecture MUST NOT change.
- Model weight SHOULD NOT change.
- New versions of trained model and inference image MUST be released.

**Pre-release version**

Pre-release version supports testing before the final release.

- Whether the model inference API, architecture, and weight would change depends on the final version that the pre-release version refers to.
- New version of trained model MUST be released.
- New version of inference image MAY be released.

**Post release** 

Post-release version indicate changes that are not distributed with the model, e.g., documentation update.

- Model inference API and architecture MUST NOT change.
- Model weight SHOULD NOT change.
- New versions of trained model and inference image MUST NOT be released.

**Developmental release**

Developmental release include experimental changes.

- Model inference API, architecture, and weight MAY change.
- New versions of trained model and inference image MAY be released.
