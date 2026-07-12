# Heavy Edge Feature Model

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/jeesoo9595/heavyedge-features-v1)
[![Documentation](https://img.shields.io/badge/github-pages-blue?logo=github)](https://heavyedge.github.io/feature-models/)
[![GitHub repository](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/heavyedge/feature-models)

Gaussian process quantile regression (GPQR) models of heavy edge shape features.

Provides:
- GPQR models for H and phi.
- Cross validation and other validation examples.
- Yield estimation and probabilistic quality window examples.

## Usage (Inference)

Pre-trained models for inference can be accessed via an image, or by installing directly from the [model repository](https://huggingface.co/jeesoo9595/heavyedge-features-v1).

### Inference image

> **NOTE**: Initial inference in a container requires a network connection to download Python packages.

Images for containerized inference are provided.

The image keeps the packaged model under `/app/model` and uses `/workdir` as the intended mount point for user input and output files.

Let the image reference be `heavyedge/feature-models:v1.0.0`, and assume the input file `X.npy` is prepared in the host directory.
The following command stores the predicted quantiles of `H` as `H.quantiles.npy` in the host directory.

```
docker run --rm -v "$PWD:/workdir" heavyedge/feature-models:v1.0.0 \
  python3 /app/model/predict-quantiles.py X.npy \
  --target H --method delta -o H.quantiles.npy
```

### Direct installation

Prepare the following prerequisites:

- [Hugging Face CLI](https://huggingface.co/docs/transformers/en/installation)
- Python runtime

Run the following commands:

```
hf download jeesoo9595/heavyedge-features-v1 --repo-type model --local-dir model
pip install -r model/requirements.txt
```

Run the `model/predict-mean.py` script to evaluate the prior mean function.
Run the `model/predict-quantiles.py` script to evaluate quantile functions.

## Usage (Training)

Models can be trained using the developer container, which includes our dataset.
The weights pushed to the model repository are trained using this container.

If you want to train the model with your own dataset, install this repository directly.

We recommend training the models in a CUDA environment.

### Training image

> **NOTE**: Initial training in a container requires a network connection to download Python packages.

Images for containerized training are provided and are identified by the `*-dev` tag.

The image keeps our dataset and repository source under the `/app` directory.

Let the image reference be `heavyedge/feature-models:v1.0.0-dev`.
The following command installs the prerequisite packages, trains the model, and stores the output under the `/app/model` directory.

```
docker run heavyedge/feature-models:v1.0.0-dev sh -c 'uv pip install --system -r requirements.txt && make models'
```

### Direct installation

Install prerequisites:

```
pip install --system -r requirements.txt
```

Download feature data (optional):

```
curl -LsSf https://hf.co/cli/install.sh | bash
hf auth login --token [Hugging Face token]
./setup.sh
```

> If you want to use your own dataset, prepare it in a CSV format that is compatible with ours.
> Then, store it as `_temp/Dataset.csv` in the project root directory, and proceed.

Train & plot performance:

```
make models
make notebooks
```

Testing built models:

```
make test
```

## Documentation

Documentation can be found at:

> https://heavyedge.github.io/feature-models/

The HTML document is also distributed in `/app/doc` directory inside the inference image.
To view the document in a browser on the host, build the image and run:

```sh
docker run --rm -p 8000:8000 heavyedge/feature-models:v1.0.0 -m http.server 8000 --directory /app/doc
```

Then open <http://localhost:8000/> in the host's browser while the container is running.

Alternatively, you can build the document by yourself.
After building the model and the notebooks, run the following commands:

```
cd doc
make html
```

The main page is `doc/build/html/index.html`.

## Contributing

### Configuring git

Configure the local git filter (run once after cloning):

```
git config filter.nbstripout.clean "nbstripout"
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true
```

### Versioning policy

This repository follows semantic versioning with [Python version specifiers](https://packaging.python.org/en/latest/specifications/version-specifiers/):

```
N.N.N[{a|b|rc}N][.postN][.devN]
```

**Major version**

A major version change indicates that a completely different model is released.
Each major version has a dedicated repository, e.g., `heavyedge-features-v1`, `heavyedge-features-v2`, and so on.

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
The trained model itself SHOULD NOT change, unless it is trained with the same data.

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
