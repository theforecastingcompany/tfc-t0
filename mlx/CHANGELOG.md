# Changelog

All notable changes to `tfc-t0-mlx` are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Versioned parity attestations and a protected, tag-bound release pipeline.
