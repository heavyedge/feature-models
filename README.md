# Heavy Edge Feature Model

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/heavyedge/feature-model-v0)
[![Documentation](https://img.shields.io/badge/github-pages-blue?logo=github)](https://heavyedge.github.io/feature-models/)
[![GitHub repository](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/heavyedge/feature-models)

Models the relation between process variables and edge shape features.

Provides:
- GPR and GPQR models for H and phi.
- Cross validation and other validation examples.
- Yield estimation and probabilistic quality window examples.

## Usage

This repository provides architectures for edge shape models, scripts to train and evaluate them, and documents to visualize the results.

### Cloning the repository

You need:

- `git`
- Python runtime with `pip`

Run the following commands to clone the repository and install the necessary requirements.

```sh
git clone git@github.com:heavyedge/feature-models.git
cd feature-models
pip install -r requirements.txt
```

### Downloading the dataset (Optional)

Run the following commands to download the process variable and shape feature datasets in the `_data` directory.

```sh
export HUGGINGFACE_TOKEN="..."
./setup.sh
```

### Acquiring the models

The classification models trained by this project can be acquired by downloading them from the model repository.
Alternatively, you can train the models yourself if you have downloaded the dataset.

Either approach creates the trained models in the `models/v*` directories.

#### Direct download

You need:

- [Hugging Face CLI](https://huggingface.co/docs/transformers/en/installation)

Run the following command:

```sh
hf download heavyedge/feature-model-v0 --repo-type model --local-dir models/v0
```

You may change the reposotiry name and local path to download other models.

## Contributing

### Configuring git

Configure the local git filter (run once after cloning):

```sh
nbstripout --install --attributes .gitattributes
git config filter.nbstripout.clean "nbstripout"
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true
```

### Building the container image

The `Dockerfile` is provided to facilitate model distribution without sharing secrets.

After downloading the dataset and training the models, build the image with one of the following targets:

- `infer`
  - Includes the trained models (`models`).
  - Includes essential environment for inference.
- `base` (default)
  - Includes the trained models (`models`).
  - Includes built examples (`examples`).
  - Includes essential environment for inference.
  - Includes non-hidden source files.
- `dev`
  - Includes the dataset (`_data`).
  - Includes the trained models (`models`).
  - Includes built examples (`examples`).
  - Includes essential environment for inference.
  - Includes all source files.

### Versioning policy

This repository follows semantic versioning with [Python version specifiers](https://packaging.python.org/en/latest/specifications/version-specifiers/):

```
N.N.N[{a|b|rc}N][.postN][.devN]
```

- Final release and pre-release (`N.N.N[{a|b|rc}N]`):
  - Models are re-trained and deployed to HuggingFace.
  - Examples are re-built using the new model and uploaded as release artifacts.
- Post-release (`*.postN`):
  - Models are pulled from the previous release and deployed to HuggingFace.
  - Examples are re-built using the pulled model and uploaded as release artifacts.
- Developmental release (`*.devN`):
  - Models are pulled from the previous release and deployed to HuggingFace.
  - Examples are re-built using the pulled model and uploaded as release artifacts.

> **NOTE**: The major version is raised when the models are changed in a backward-incompatible way.
> When the major version is raised, trained models are deployed in the new repository.
> For example, `models/v0` is uploaded to `heavyedge-features-v0`, `models/v1` to `heavyedge-classify-v1`, and so on.
