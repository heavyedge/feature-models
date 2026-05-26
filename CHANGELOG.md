# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## UNRELEASED

### Fixed

- Direct GPQRs now uses correct likelihoods.
  This change reveals that direct GPQRs are actually better than center-gap GPQRs.

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