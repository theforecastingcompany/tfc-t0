# Changelog

All notable changes to `tfc-t0-mlx` are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0a1] - 2026-09-05

### Fixed

- Preserve annotated release tags during source verification and build from
  the verified commit. The `0.1.0a0` publication stopped at source verification;
  this release contains the same runtime implementation.
- Document the full FP32 setting used for checkpoint parity on Apple M5.

## [0.1.0a0] - 2026-09-04

### Added

- First-party, inference-only MLX implementation of `t0-alpha`.
- Direct loading of the original `config.json` and `model.safetensors` without
  rewriting weights.
- Univariate and multivariate forecasting, typed masks, explicit grouping,
  known-future covariates, quantile interpolation and autoregressive rollout.
- Opt-in compilation for repeated input shapes.
- Checkpoint-backed FP32 parity coverage against the first-party PyTorch
  implementation.
- Reproducible comparisons with PyTorch MPS and CPU on Apple silicon.
- Versioned parity attestations and a tag-bound release pipeline.
