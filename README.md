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
git clone git@github.com:heavyedge/feature-models.git
cd feature-models
pip install -r requirements.txt
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

#### Base image

Base image includes source code, documents and trained models.
The base images are tagged by `(version)`:

```sh
docker pull jeesoo9595/heavyedge-feature-models:latest
```

After pulling the image, run the following command.
It creates a container named `feature-models`, downloads dependencies and attaches to the terminal.

```sh
docker run --name feature-models -it jeesoo9595/heavyedge-feature-models:latest sh -c "uv pip install --system -r requirements.txt && /bin/bash"
```

#### Inference image

Inference image is the minimum installation, including only the trained models.
The inference images are tagged by `(version)-infer`:

```sh
docker pull jeesoo9595/heavyedge-feature-models:latest-infer
```

After pulling the image, run the following command.
It creates a container named `feature-models`, downloads dependencies and attaches to the terminal.

```sh
docker run --name feature-models -it jeesoo9595/heavyedge-feature-models:latest-infer sh -c "uv pip install --system -r model/requirements.txt && /bin/bash"
```

## Usage

You can use this project in three ways:

1. Train the model using your own data.
2. Do inference using the trained model.
3. Use advanced features, e.g., joint probability estimation.

### Training (optional)

> This feature is accessible if you
> - Cloned the repository, or
> - Pulled the base image.

To train your own model, run scripts in `scripts/train/` directory.
You need to prepare your own dataset as csv files and pass them to the scripts.
Refer to the recipes in `Makefile`.

Alternatively, you can use the distributed pre-trained model.

### Inference

> This feature is accessible if you
> - Cloned the repository, or
> - Downloaded from the model repository, or
> - Pulled the base image, or
> - Pulled the inference image.

For inference, you need to put trained models and inference scripts in `model/` directory.
These files are already provided if you downloaded from the model repository or pulled the image.

To perform inference, run scripts in `model/` directory.
You need to prepare your own input data points as npy file and pass it to the script.
Refer to the recipes in `Makefile`.

### Advanced features

> This feature is accessible if you
> - Cloned the repository, or
> - Pulled the base image.

Files in `scripts/` and `notebooks/` directories provide advanced features.
These features include models with different architectures, model selection, and joint probability estimation.
Refer to the recipes in `Makefile` and project document.

> **NOTE** : Advanced features are considered experimental.
> Backward-incompatible changes may occur without versioning support.

## Documentation

Documentation can be found at:

> https://heavyedge.github.io/feature-models/

The HTML document is also distributed in `doc/` directory inside the image.
To view the document in a browser on the host, build the image and run:

```sh
docker run --rm -p 8000:8000 jeesoo9595/heavyedge-feature-models:latest python -m http.server 8000 --directory ./doc
```

Then open <http://localhost:8000/> in the host's browser while the container is running.

## Contributing

### Configuring git

Configure the local git filter (run once after cloning):

```sh
git config filter.nbstripout.clean "nbstripout"
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true
```

### Downloading the dataset

The default dataset can be downloaded by running the following command with Huggingface credential.
This dataset is used to train the models that are published to the model repository.

```sh
hf auth login --token [Hugging Face token]
./download.sh
```

### Building the model

```sh
make models
```

### Testing the built model

```sh
make test
```

### Building the document

After building the models, run the following commands:

```sh
make notbooks
pip install -r doc/requirements.txt
cd doc
make html
```

The main page is `doc/build/html/index.html`.

### Development image

The development image includes dataset and additonal files for containerized training.
It can be built by running the `Dockerfile` with `dev` taget, passing the `hf_token` secret.

The development images are tagged by `(version)-dev`, but they are never released.
They are uploaded to private registry during CI/CD and immediately deleted afterwards.

### Versioning policy

This repository follows semantic versioning with [Python version specifiers](https://packaging.python.org/en/latest/specifications/version-specifiers/):

```
N.N.N[{a|b|rc}N][.postN][.devN]
```

The repository versioning focuses on managing versions of trained models.
If only the repository code is updated without changing the model, the following rules apply:

- `notebooks` : Treated as documentation updates, so a post-release is published.
- `scripts` : Treated as developer features, so a developmental release is published.

Trained models are released only when the final relase or the pre-release are made.
Inference image is released only when new trained models are released.
Base image is always released when a new release is made in the repository.

#### Major version

A major version change indicates that a new model with incompatible API is released.
Each major version has a dedicated repository, e.g., `heavyedge-features-v0`, `heavyedge-features-v1`, and so on.

- Model API MAY change.
- Model architecture MAY change.
- Model weights MAY change.

#### Minor version

A minor version change indicates that a new model with compatible API is released.
This can be a change in architecture or training dataset.

- Model API MUST NOT change.
- Model architecture MAY change.
- Model weights MAY change.

#### Patch version

A patch version change indicates a change in hyperparameter or bug fix.

- Model API MUST NOT change.
- Model architecture MUST NOT change.
- Model weights MAY change.

#### Pre-release version

A pre-release version is for testing before the final release.
The scope and extent of model change depends on the final version that the pre-release version refers to.

#### Post-release version

Post-release versions indicate minor changes that do not affect the software, e.g., documentation updates.
Model MUST NOT change in any way.

#### Developmental release

Developmental releases include experimental changes in the advanced features.
Model MUST NOT change in any way.
