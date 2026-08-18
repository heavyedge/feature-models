# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0rc1] - 2026-08-18

- Model: `gpytorch-qr==0.9.0`
- Dataset:
  - `heavyedge/profiles:v1.0.0`
  - `heavyedge/shape-features:v1.0.0b1`

### Fixed

- Optuna now catches `torch.LinAlgError`.

## [0.3.0rc1] - 2026-08-18

- Model: `gpytorch-qr==0.9.0`
- Dataset:
  - `heavyedge/profiles:v1.0.0`
  - `heavyedge/shape-features:v1.0.0b1`

### Changed

- Default GPQR model now uses independent multitask latent functions.

### Fixed

- Optuna now catches `torch.LinAlgError`.

## [1.0.0rc0] - 2026-08-17

- Model: `gpytorch-qr==0.9.0rc0`
- Dataset:
  - `heavyedge/profiles:v1.0.0rc4`
  - `heavyedge/shape-features:v1.0.0b1`

### Added

- GPQR models are added.

- `examples/v1/CV.ipynb` is added.
- `examples/v1/Window.ipynb` is added.

### Changed

- GPR models are now `ApproximateGP` with inducing points.

## [0.3.0rc0] - 2026-08-16

- Model: `gpytorch-qr==0.9.0rc0`
- Dataset:
  - `heavyedge/profiles:v1.0.0rc4`
  - `heavyedge/shape-features:v1.0.0b1`

### Added

- `train/gpqr.py` now has `--num-likelihood-samples` argument.
- Direct GPQR with independent multitask variational strategy is added.
- Center-gap GPQR with independent multitask variational strategy is added.
- Center-gap GPQR with standard LMC multitask variational strategy is added.

### Changed

- Direct GPQR is renamed to explicitly specify LMC variational strategy.
- Center-gap GPQR is renamed to explicitly specify center-gap LMC variational strategy.

### Fixed

- `train/gpqr.py` now removes duplicate inducing points.

## [0.3.0.dev1] - 2026-08-15

- Model: `gpytorch-qr==0.9.0rc0`
- Dataset:
  - `heavyedge/profiles:v1.0.0rc4`
  - `heavyedge/shape-features:v1.0.0b1`

### Changed

- GPR and GPQR no longer has lengthscale constraint.
- GPR and GPQR now use hyperparameter optimization to choose noise prior and lengthscale prior.

### Removed

- Extrapolation validation is removed.

## [1.0.0a3] - 2026-08-12

- Model: `gpytorch-qr==0.8.0`
- Dataset:
  - `heavyedge/profiles:v1.0.0rc4`
  - `heavyedge/shape-features:v1.0.0a4`

### Changed

- GPR now uses train data + validation data for final training.
- Increase early stopping patience ratio of GPR from 0.02 to 0.1
- Increase pruning patience ratio of GPR from 0.02 to 0.1
- Increase the number of hyperparameter optimization trial from 50 to 100

## [1.0.0a2] - 2026-08-06

- Model: `gpytorch-qr==0.8.0`
- Dataset:
  - `heavyedge/profiles:v1.0.0rc4`
  - `heavyedge/shape-features:v1.0.0a4`

### Changed

- Implement pruning for GPR of `H` and `phi`.
- Make early stopping more patient for GPR of `H` and `phi`.

## [1.0.0a1] - 2026-08-06

- Model: `gpytorch-qr==0.8.0`
- Dataset:
  - `heavyedge/profiles:v1.0.0rc4`
  - `heavyedge/shape-features:v1.0.0a4`

### Added

- v1 GPR of `H` and `phi`.

## [1.0.0a0] - 2026-08-05

- Model: `gpytorch-qr==0.8.0`
- Dataset:
  - `heavyedge/profiles:v1.0.0rc4`
  - `heavyedge/shape-features:v1.0.0a4`

### Added

- v1 Prior mean of `H` and `phi`.

## [0.3.0.dev0] - 2026-08-12

- Model: `gpytorch-qr==0.8.0`
- Dataset:
  - `heavyedge/profiles:v1.0.0rc4`
  - `heavyedge/shape-features:v1.0.0a4`

### Changed

- Use `heavyedge/profiles:v1.0.0rc4` for process variable data.

### Fixed

- `__pycache__` is no longer pushed to HuggingFace.

## [0.2.0] - 2026-08-04

- Model: `gpytorch-qr==0.8.0`
- Dataset:
  - `heavyedge/profiles:v1.0.0rc3`
  - `heavyedge/shape-features:v1.0.0a4`

### Changed

- Use `heavyedge/profiles:v1.0.0rc3` for process variable data.
- Use `heavyedge/shape-features:v1.0.0a4` for shape feature data.
- Model repository is changed to `heavyedge/feature-model`.

### Removed

- HTML documentation is removed since it is redundant.

## [0.1.0] - 2026-07-19

Trained with:

- Model: `gpytorch-qr==0.8.0`
- Dataset: `jeesoo9595/heavyedge-features-v1:v1.3.0`

### Added

- v0 Prior mean of `H` and `phi`.
- v0 GPR of `H` and `phi`.
- v0 GPQR of `H` and `phi`.
