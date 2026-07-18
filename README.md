# Heavy Edge Feature Model

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/jeesoo9595/heavyedge-features-v0)
[![Docker](https://img.shields.io/badge/-Docker-2496ED?style=flat-square&logo=Docker&logoColor=white)](https://hub.docker.com/repository/docker/jeesoo9595/heavyedge-feature-models)
[![Documentation](https://img.shields.io/badge/github-pages-blue?logo=github)](https://heavyedge.github.io/feature-models/)
[![GitHub repository](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/heavyedge/feature-models)

Gaussian process quantile regression (GPQR) models of heavy edge shape features.

Provides:
- GPQR models for H and phi.
- Cross validation and other validation examples.
- Yield estimation and probabilistic quality window examples.

## Installation

This repository provides model architectures, and scripts to train and evaluate the model.
Clone the repository and execute the files in `scripts` directory.
Refer to the `Makefile` for usage examples.

The trained model can be acquired by either directly downloading from the [model repository](https://huggingface.co/jeesoo9595/heavyedge-features-v0) of by pulling the Docker image that encapsulates it.

### Cloning the repository

```sh
git@github.com:heavyedge/feature-models.git
```

### Direct download

- [Hugging Face CLI](https://huggingface.co/docs/transformers/en/installation)
- Python runtime

Run the following commands:

```sh
hf download jeesoo9595/heavyedge-features-v0 --repo-type model --local-dir model
pip install -r model/requirements.txt
```

### Docker image

Two types of images are distributed for each release: the base image and the inference image.

Base image includes source code, documents and trained models.
The base images are tagged by `(version)`:

```sh
docker pull jeesoo9595/heavyedge-feature-models:latest
```

Inference image is the minimum installation, including only the trained models.
The inference images are tagged by `(version)-infer`:

```sh
docker pull jeesoo9595/heavyedge-feature-models:latest-infer
```

## Usage

You can use this project in three ways:

1. Train the model using your own data.
2. Do inference using the trained model.
3. Perform advanced inference, e.g., joint probability estimation.

### Training

> This feature is accessible if you
> - Cloned the repository, or
> - Pulled the base image.

### Inference

> This feature is accessible if you
> - Cloned the repository, or
> - Downloaded from the model repository, or
> - Pulled the base image, or
> - Pulled the inference image.

### Advanced inference

## Contributing

### Configuring git

Configure the local git filter (run once after cloning):

```
git config filter.nbstripout.clean "nbstripout"
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true
```

### Building the document

After building the model and the notebooks, run the following commands:

```
cd doc
make html
```

The main page is `doc/build/html/index.html`.

### Versioning policy

This repository follows semantic versioning with [Python version specifiers](https://packaging.python.org/en/latest/specifications/version-specifiers/):

```
N.N.N[{a|b|rc}N][.postN][.devN]
```

**Major version**

A major version change indicates that a completely different model is released.
Each major version has a dedicated repository, e.g., `heavyedge-features-v0`, `heavyedge-features-v2`, and so on.

- Model inference API MAY change.
- Model architecture and weights MAY change.
- New versions of trained model and inference image MUST be released.

**Minor version**

A minor version change indicates that the same model is re-trained with different data and released.

- Model inference API and architecture MUST NOT change.
- Model weights MAY change.
- New versions of trained model and inference image MUST be released.

**Patch version**

A patch version change indicates metadata changes and bug fixes in auxiliary scripts that are distributed with the model.
The trained model itself SHOULD NOT change, unless it is retrained with the same data.

- Model inference API and architecture MUST NOT change.
- Model weights SHOULD NOT change.
- New versions of trained model and inference image MUST be released.

**Pre-release version**

A pre-release version supports testing before the final release.

- Whether the model inference API, architecture, and weights change depends on the final version that the pre-release version refers to.
- A new version of the trained model MUST be released.
- A new version of the inference image MAY be released.

**Post-release version** 

Post-release versions indicate changes that are not distributed with the model, e.g., documentation updates.

- Model inference API and architecture MUST NOT change.
- Model weights SHOULD NOT change.
- New versions of trained model and inference image MUST NOT be released.

**Developmental release**

Developmental releases include experimental changes.

- Model inference API, architecture, and weights MAY change.
- New versions of trained model and inference image MAY be released.
