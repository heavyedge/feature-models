# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [UNRELEASED]

- Model: `gpytorch-qr==0.11.0`
- Dataset:
  - `heavyedge/profiles:v1.0.0`
  - `heavyedge/shape-features:v1.1.0`

### Added

- Prediction scripts now print logs.
- `Models.prior_mean.ipynb` is added.
- `Models.gpr.ipynb` is added.
- `Models.gpqr.ipynb` is added.

### Changed

- Joint probability estimation now uses CUDA.
- Threshold for `phi` is set to `0.25` from `0.5`.
- Data are splitted and redrawn for unique X.
- Quantile interpolation now uses spline with exponential tail model.
- Hyperparameters are optimized by 5-fold cross validation.
- GPQR now use GPR posterior mean as prior mean.
- GPQR now use GPR lengthscale.
- GPQR predictions now enforce strict quantile ordering without dtype promotion.

### Removed

- `Models.ipynb` is removed.

## [1.0.0rc6] - 2026-08-23

- Model: `gpytorch-qr==0.9.0`
- Dataset:
  - `heavyedge/profiles:v1.0.0`
  - `heavyedge/shape-features:v1.1.0`

### Added

- Prediction scripts now print logs.
- Class probability example is added.

### Changed

- Joint probability estimation now uses CUDA.
- Threshold for `phi` is set to `0.25` from `0.5`.

## [1.0.0rc5] - 2026-08-23

- Model: `gpytorch-qr==0.9.0`
- Dataset:
  - `heavyedge/profiles:v1.0.0`
  - `heavyedge/shape-features:v1.1.0`

### Changed

- Joint probability estimation now uses CUDA.
- Threshold for `phi` is set to `0.5`.

### Fixed

- Joint probability now again depends on marginal probabilities of `H` and `phi_1`.

## [1.0.0rc4] - 2026-08-22

- Model: `gpytorch-qr==0.9.0`
- Dataset:
  - `heavyedge/profiles:v1.0.0`
  - `heavyedge/shape-features:v1.1.0`

### Changed

- v1 now models `H`, `phi_1`, and `phi_3` as three independent GP batches in
  consolidated `prior_mean.pt`, `gpr.pt`, and `gpqr.pt` artifacts.
- Noise prior hyperparameter is no longer optimized.
- Latent function number hyperparameter is no longer optimized.
- Number of HPO trial is reduced to 10 from the previous value of 100.

### Fixed

- Parallel model builds now serialize SQLite Optuna storage initialization to
  avoid concurrent schema creation failures.
- Jitter value is increased to avoid PSD error.

### Fixed

- `linear_operator.utils.errors.NotPSDError` is now catched during training.

## [1.0.0rc3] - 2026-08-20

- Model: `gpytorch-qr==0.9.0`
- Dataset:
  - `heavyedge/profiles:v1.0.0`
  - `heavyedge/shape-features:v1.1.0`

### Fixed

- `linear_operator.utils.errors.NotPSDError` is now catched during training.

## [1.0.0rc2] - 2026-08-19

- Model: `gpytorch-qr==0.9.0`
- Dataset:
  - `heavyedge/profiles:v1.0.0`
  - `heavyedge/shape-features:v1.1.0`

### Changed

- `predict-gpr.py` now returns mean and std for both latent posterior and predictive posterior.

## [1.0.0rc1] - 2026-08-18

- Model: `gpytorch-qr==0.9.0`
- Dataset:
  - `heavyedge/profiles:v1.0.0`
  - `heavyedge/shape-features:v1.0.0b1`

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
