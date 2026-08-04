# Heavy Edge Feature Model

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/heavyedge/feature-model-v0)
[![GitHub repository](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/heavyedge/feature-models)

Models the relation between process variables and edge shape features.

Provides:
- GPR and GPQR models for edge shape features.
- Cross validation and other benchmarks.
- Yield estimation and probabilistic quality window examples.

## Usage

This repository provides architectures for edge shape models, scripts to train and evaluate them, and notebooks to visualize the results.

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

The models trained by this project can be acquired by downloading them from the model repository.
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

#### Training the models

You need:

- `make`

Run the following command:

```sh
make models
```

You can test the trained models by running:

```sh
make tests
```

### Using the trained model

Once models are trained, you can perform inference using the scripts in `models/v*` directory.

### Acquiring the built examples

The benchmark results are visualized as notebooks in the `examples` directory.

The notebook outputs are stripped before being stored in this repository.
To check their outputs, you must acquire the built example notebooks.

You can either download the built notebooks from the [GitHub release](https://github.com/heavyedge/feature-models/releases) artifacts, or build the notebooks yourself if you have acquired the preprocessed data.

#### Building the notebooks

You need:

- `make`

```sh
pip install -r examples/requirements.txt
make examples
```

## Contributing

### Configuring git

Configure the local git filter (run once after cloning):

```sh
nbstripout --install --attributes .gitattributes
git config filter.nbstripout.clean "nbstripout"
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true
```

### Testing

Set the `HEAVYEDGE_TEST_MODE` environment variable to `1` for CI/CD testing purposes.

```sh
export HEAVYEDGE_TEST_MODE=1
./setup.sh
make models examples tests
```

### Building the container image

The `Dockerfile` is provided to facilitate model distribution without sharing secrets.

After downloading the dataset and training the models, build the image with one of the following targets:

- `infer`
  - Includes the trained models (`models`).
  - Includes essential environment for inference.
- `base` (default)
  - Includes the trained models (`models`).
  - Includes the benchmarks and built examples (`benchmarks`, `examples`).
  - Includes essential environment for inference.
  - Includes non-hidden source files.
- `dev`
  - Includes the dataset (`_data`).
  - Includes the trained models (`models`).
  - Includes the benchmarks and built examples (`benchmarks`, `examples`).
  - Includes essential environment for inference.
  - Includes all source files.

### Versioning policy

This repository follows semantic versioning with [Python version specifiers](https://packaging.python.org/en/latest/specifications/version-specifiers/):

```
N.N.N[{a|b|rc}N][.postN][.devN]
```

The following rules apply to the final release versions:

- Major version is raised when the model API is changed in a backwards-incompatible way.
- Minor version is raised when the models are trained with new dataset.
- Patch version is raised when bugs are fixed.

This repository stores source code for all major versions.
When a new release is made, trained models are deployed to a repository dedicated to each major release.
For example, when `v1.0.0` is released, `models/v1` is uploaded to `feature-model-v1` repository.
This applies to other build outputs, e.g., `examples/v1`.
