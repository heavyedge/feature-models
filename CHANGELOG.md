# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0.a0] - 2026-08-05

- Model: `gpytorch-qr==0.8.0`
- Dataset:
  - `heavyedge/profiles:v1.0.0rc4`
  - `heavyedge/shape-features:v1.0.0a4`

### Added

- Prior mean of `H` and `phi`.

## [0.2.0.post0] - 2026-08-05

- Model: `gpytorch-qr==0.8.0`
- Dataset:
  - `heavyedge/profiles:v1.0.0rc4`
  - `heavyedge/shape-features:v1.0.0a4`

### Changed

- Use `heavyedge/profiles:v1.0.0rc4` for process variable data.

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

- Prior mean of `H` and `phi`.
- GPR of `H` and `phi`.
- GPQR of `H` and `phi`.
