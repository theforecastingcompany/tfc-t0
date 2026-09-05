# Changelog

All notable changes to `tfc-t0` are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Preserve Hugging Face configuration-download errors in `from_pretrained`,
  including gated-model access failures, instead of reporting missing model
  constructor arguments.
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
