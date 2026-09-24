# Changelog

All notable changes to `tfc-t0-mlx` are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The README now links to the `t0` paper on arXiv
  ([2609.24559](https://arxiv.org/abs/2609.24559)) with a badge, and its
  citation block cites the paper.

### Fixed

- Variate attention now matches training when known-future covariates are
  present: a future-covariate row reads only future-covariate rows, so
  information flows from covariates into targets and never back. Forecasts
  without future covariates are bit-identical; forecasts with them change.
  Mirrors the same fix in `tfc-t0`.
- A patch now takes its variate type from its first non-padding time step, so
  a left-padded first patch on a covariate row is no longer embedded as a
  target. Mirrors the same fix in `tfc-t0`.

### Changed

- **Breaking.** `T0Forecaster.predict`'s `quantiles` argument is now
  `quantile_levels`. It always carried levels rather than quantile values, and
  the returned `Forecast` already called them `quantile_levels`, so the same
  concept had two names on either side of one call. Replace
  `predict(..., quantiles=[0.1, 0.5, 0.9])` with
  `predict(..., quantile_levels=[0.1, 0.5, 0.9])`. `Forecast.quantiles`, which
  holds the forecast values, is unchanged.

- Quantile levels requested beyond the trained range (outside [0.1, 0.9]) now
  follow IQF exponential tails
  ([Park et al., arXiv 2111.06581](https://arxiv.org/abs/2111.06581)) pinned
  through the outermost trained levels, instead of clamping flat to the nearest
  trained level. A requested tail level no longer costs an autoregressive rollout
  path of its own. Requesting one does add the trained levels it pins through to
  the rollout, so over horizons long enough to roll out, the other requested
  levels can shift slightly.

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
