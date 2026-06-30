# Changelog

All notable changes to `tfc-t0` are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
