# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - UNRELEASED

### Changed

- H GPQR model now uses center-gap to prevent quantile crossing.

## [1.3.0] - 2026-05-28

### Added

- Function and notebooks to test quantile crossing.

### Changed

- H GPQR model is changed to prevent quantile crossing.
- phi GPQR model is changed to prevent quantile crossing.

## [1.2.0] - 2026-05-27

### Fixed

- Direct GPQRs now uses correct likelihoods.
- Direct GPQRs now have correct offsets.

### Changed

- GPQR models now use direct GPQR.

### Deprecated

- `predict.py` is re-added and deprecated.

## [1.1.0] - 2026-05-25

### Changed

- `predict.py` is changed to `load.py`.

## [1.0.0] - 2026-05-24

Trained with:

- Model: `gpytorch-qr>=0.5.0,<0.6.0`
- Dataset: `jeesoo9595/heavyedge-features-v1:v1.3.0`