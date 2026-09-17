# Changelog

All notable changes to `tfc-t0` are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-09-17

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
  trained level.

## [0.4.0] - 2026-09-15

Breaking. `T0Forecaster` now has two forecasting methods instead of four.

### Changed
- `forward(model_input)` runs a single differentiable pass over a `TimeSeries`, with
  no rollout. Use it to fine-tune.
- `predict(model_input, horizon, quantiles, ...)` is inference-only and rolls out
  autoregressively past `max_horizon`. It takes a `TimeSeries`, and still converts a
  raw context array on the fly, so array callers are unaffected.
- `context_length` is optional and defaults to the start of the forecast region.
- `prepare_rollout_buffer` and `predict_step` are internal to `RolloutManager` again.

### Removed
- `predict_from_time_series`. Replace `predict_from_time_series(ts, h, ctx, q)` with
  `predict(ts, h, q, context_length=ctx)`.

### Fixed
- `Patcher.pad` aligned the patch grid to the input's total width rather than to the
  context boundary. It now left-pads to the context boundary and right-pads the
  forecast region to whole patches. `predict` was never affected, because
  `RolloutManager` already aligned its buffer. Calling `forward` directly with
  known-future covariates returned a wrong forecast that still looked valid. On
  `t0-alpha` with a 24-step context and a 12-step horizon it moved MAE from 3.72 to 8.32.

### Added
- `horizon` on `TimeSeries.from_array`, extending the target rows with that many
  `WITHHELD` steps to mark the region to predict. Future covariates imply it from
  their width.
- `Patcher.context_end`, the index where the forecast region begins.
- `time_series_from_array` in `t0.data`, to build a `TimeSeries` from a raw context array.
- Gradient tests in `tests/test_forward.py` and doctests on `Patcher.pad`.

## [0.3.2] - 2026-09-09

### Added
- `TimeSeries.batch` and `T0Forecaster.predict_from_time_series` for integrations
  that construct complete T0 inputs, including known-future covariates.

### Fixed
- Document the model-access and authentication steps before the PyTorch
  quickstart in the README and model card.

## [0.3.1] - 2026-09-01

Documentation-only release.

### Fixed
- Image paths in README
- Hugging Face license links in the model card.

## [0.3.0] - 2026-08-19

### Added
- `batch_series` — batch series of different lengths and variate counts into one
  padded context, the mask describing it, and the group ids that keep each
  series' variates forecast jointly. Pass all three to `predict`.
- `mask` and `group_ids` arguments on `T0Forecaster.predict`. `mask` marks every
  context cell as an absent observation (`MaskType.MISSING`) or as padding for a
  shorter series (`MaskType.PAD`); `group_ids` says which context rows are
  variates of one series. `MaskType` is exported for building masks by hand.
- Without a `mask`, NaN in the context is read as a missing observation, as
  before — padding is the one case NaN cannot express on its own.

### Fixed
- Padding a batch of unequal-length series no longer moves the shorter rows'
  forecasts. A patch covered entirely by padding drops out of attention, so a
  series' forecast no longer depends on how long its batch mates are — it moved
  by up to 12.5% before. Batch through `batch_series`, or mark the padding
  `MaskType.PAD`, to get it.

## [0.2.3] - 2026-07-30

### Fixed
- First patches were not attended to when leading time steps were padding.
It resulted in degraded performances of the model.

### Changed
- Internals of the model regarding mask computation and patched variables handling.

## [0.2.2] - 2026-07-20

### Changed
- Removed the upper version caps on all core runtime dependencies (`torch`,
  `einops`, `rotary-embedding-torch`, `huggingface-hub`, `safetensors`,
  `numpy`, `jaxtyping`). Constraints are now floor-only, so downstream
  integrators (e.g. Darts) can resolve newer releases without being blocked.
  Floors are bumped when a new major is tested rather than capped speculatively.
- Removed the upper version caps on the optional extras as well (`gluonts`,
  `matplotlib`, `pandas`), matching the floor-only policy.
- Dropped the `requires-python` upper bound (`<3.15`); the package now declares
  `>=3.10` with no ceiling, so it never blocks users on newer Python releases.

## [0.2.1] - 2026-06-26

### Added
- Python 3.10 support — the minimum supported version is now 3.10 (previously 3.11).

### Fixed
- Inference quickstart notebook: move the median forecast to CPU before
  computing the error metric, so the dtype-routing demo runs on GPU as well as
  on CPU.

## [0.2.0] - 2026-06-24

### Added
- bf16/fp16 mixed-precision inference: pass `dtype=torch.bfloat16` (or
  `torch.float16`) to `from_pretrained` or the constructor — weights stay fp32
  and the forward pass is autocast.
- Example notebooks: an inference quickstart and a LoRA fine-tuning walkthrough,
  with a `notebooks` extra that installs everything needed to run them.

## [0.1.2] - 2026-06-14

### Changed
- Relaxed the `einops` and `jaxtyping` lower bounds so `tfc-t0` can be installed
  alongside packages that pin older versions of them.

## [0.1.1] - 2026-06-11

### Added
- Hugging Face Hub metadata on the model (pipeline tag, license, tags, and
  source repository URL) so the model page renders richer metadata.

## [0.1.0] - 2026-06-09

### Added
- Initial public release of the open-weights t0-alpha forecasting model.
