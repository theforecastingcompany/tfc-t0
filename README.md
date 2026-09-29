<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://www.theforecastingcompany.com/logo/logo_horizontal_dark.png" />
    <img src="https://www.theforecastingcompany.com/logo/logo_horizontal_light.png" alt="The Forecasting Company" width="280" />
  </picture>
</p>

---

<p align="center">
  <a href="https://arxiv.org/abs/2609.24559"><img src="https://img.shields.io/static/v1?label=t0-Report&amp;message=2609.24559&amp;color=B31B1B&amp;logo=arXiv" alt="t0 report on arXiv: 2609.24559" /></a>
  <a href="https://huggingface.co/collections/theforecastingcompany/t0-model-family"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Collection-FFD21E" alt="Hugging Face t0 model collection" /></a>
  <a href="https://colab.research.google.com/github/theforecastingcompany/tfc-t0/blob/main/notebooks/01_inference_quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open inference quickstart in Colab" /></a>
  <a href="https://pypi.org/project/tfc-t0/"><img src="https://img.shields.io/pypi/v/tfc-t0" alt="PyPI" /></a>
  <a href="https://pypi.org/project/tfc-t0/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+" /></a>
  <a href="https://github.com/theforecastingcompany/tfc-t0/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/tfc-t0" alt="License" /></a>
</p>

This repository contains the code to run `t0`, an open-weights time-series forecasting foundation model from [The Forecasting Company](https://theforecastingcompany.com/) that produces probabilistic forecasts using historical and known-future covariates.

## Contents

- [Forecasting](#-forecasting)
- [Quickstart](#-quickstart)
  - [Forecasting with covariates](#forecasting-with-covariates)
  - [Batched inference](#batched-inference)
  - [Converting your data to TimeSeries](#converting-your-data-to-timeseries)
- [Choose how to run t0 models](#choose-how-to-run-t0-models)
- [MLX runtime (Apple silicon)](#mlx-runtime-apple-silicon)
- [Architecture](#️-architecture)
  - [Lineage](#-lineage)
- [Citation](#-citation)
- [License](#️-license)

## 📈 Forecasting

## Choose how to run `t0-alpha`

This package is the first-party PyTorch runtime. The same original checkpoint
is also available through a first-party MLX runtime and our managed API:

| Use case | Install or open |
| --- | --- |
| Local inference with PyTorch | `pip install tfc-t0` |
| Local inference on Apple silicon with MLX | [`pip install tfc-t0-mlx`](https://pypi.org/project/tfc-t0-mlx/) |
| Managed inference without local weights | [The Forecasting Company API](https://docs.retrocast.com/documentation/t0-alpha) |

The MLX runtime is inference-only, has a similar `T0Forecaster.predict()` API,
loads this model's safetensors directly and does not install PyTorch.

![t0 forecasting French national electricity demand in Retrocast](https://raw.githubusercontent.com/theforecastingcompany/tfc-t0/main/assets/enedis_with_holidays.webp)

_`t0` forecasting French national electricity demand in Retrocast. Data:
[Enedis open data](https://data.enedis.fr/)._

`t0` can use past and known-future covariates to improve its forecasts:

| Without covariates                                                                                                                   | With covariates                                                                                                                |
| ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| ![t0 forecast without covariates](https://raw.githubusercontent.com/theforecastingcompany/tfc-t0/main/assets/medicam_without_cov.webp) | ![t0 forecast with covariates](https://raw.githubusercontent.com/theforecastingcompany/tfc-t0/main/assets/medicam_with_cov.webp) |

_Data: [Medic'AM](https://www.assurance-maladie.ameli.fr/etudes-et-donnees/medicaments-classe-atc-medicam),
monthly drug reimbursements from the French national health insurance._

For a complete worked example, the
[European day-ahead electricity prices notebook](https://github.com/theforecastingcompany/tfc-t0/blob/main/notebooks/03_electricity_day_ahead_prices.ipynb)
forecasts hourly prices in four bidding zones with load, renewable, holiday
and weather covariates, and backtests the result.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theforecastingcompany/tfc-t0/blob/main/notebooks/03_electricity_day_ahead_prices.ipynb)

## 🚀 Quickstart

```bash
pip install "tfc-t0[torch]>=0.5.0"
```

`t0-beta` normalizes its inputs differently from `t0-alpha`, and carries the
convention in `config.json` as `scaler_eps` and `scaler_eps_mode`. Releases
before 0.5.0 do not read those fields: they load these weights without error
and run them under `t0-alpha`'s normalization, which silently degrades the
forecast. Pin the floor.

The simplest path is a univariate forecast through `predict`:

```python
import torch
from t0 import T0Forecaster

model = T0Forecaster.from_pretrained("theforecastingcompany/t0-beta").eval()

context = torch.randn(4, 512)  # 4 series, 512 past timesteps
out = model.predict(context, horizon=64, quantile_levels=[0.1, 0.5, 0.9])
out.quantiles  # (4, 64, 3)
out.median     # (4, 64)
```

`predict` accepts `numpy` arrays. 1-D contexts are auto-promoted to a
single-row batch. NaN in the context is read as a missing observation. To
say that some cells are padding instead, pass a `mask`. See
[batched inference](#batched-inference).

`t0-alpha` and `t0-beta` are public and download without authentication.

### Forecasting with covariates

Anything you know over the **past** goes in `context` — alongside the
target, extra variates attend to it and are forecast together. Anything
you know over the **future** (calendar features, planned promotions,
weather forecasts) goes in `future_covariates`, shaped
`[B, F, context + horizon]`. The model conditions on it but does not
forecast it.

```python
import torch
from t0 import T0Forecaster

model = T0Forecaster.from_pretrained("theforecastingcompany/t0-beta").eval()

context = torch.randn(2, 512)                    # 2 series, 512 past timesteps
future_covariates = torch.randn(2, 3, 512 + 64)  # 3 covariates known over context + horizon

out = model.predict(
    context,
    horizon=64,
    quantile_levels=[0.1, 0.5, 0.9],
    future_covariates=future_covariates,
)
out.quantiles  # (2, 64, 3)
out.median     # (2, 64)
```

### Batched inference

```python
import numpy as np
from t0 import T0Forecaster, batch_series

model = T0Forecaster.from_pretrained("theforecastingcompany/t0-beta").eval()

daily = np.random.randn(180)    # one series, 180 past timesteps
store = np.random.randn(2, 96)  # one series of 2 variates, 96 past timesteps
hourly = np.random.randn(1024)  # one series, 1024 past timesteps

context, mask, group_ids = batch_series([daily, store, hourly])
context.shape  # (4, 1024) — variates stacked, right-aligned to the longest
group_ids      # [0, 1, 1, 2] — `store`'s two variates are forecast jointly

out = model.predict(
    context,
    horizon=24,
    quantile_levels=[0.1, 0.5, 0.9],
    mask=mask,
    group_ids=group_ids,
)
out.quantiles  # (4, 24, 3)
out.median[0]  # the 24-step median forecast for `daily`
```

Integrations that prepare complete T0 inputs, including known-future
covariates, can batch the native representation directly:

```python
from t0 import TimeSeries

first = TimeSeries.from_array(context_1, future_covariates_1)
second = TimeSeries.from_array(context_2, future_covariates_2)
batch = TimeSeries.batch([first, second])

out = model.predict(
    batch,
    horizon=64,
    context_length=max(context_1.shape[-1], context_2.shape[-1]),
)
```

Here each context includes its batch axis, for example `[1, V, T]`, and each
known-future input is `[1, F, T + horizon]`. The output is ordered by the
flattened target rows in `batch`.

### Converting your data to `TimeSeries`

`TimeSeries` is the model's native input. It holds target rows, known-future
covariate rows, a mask and group ids, all on one width. `predict` builds one for
you from a raw array. You only need to construct one yourself to batch inputs of
different widths, or to call `forward` directly.

```python
from t0 import TimeSeries

# context only, with `horizon` marking the region to predict
model_input = TimeSeries.from_array(context, horizon=24)          # context: [B, V, T]

# with known-future covariates, whose width sets the horizon
model_input = TimeSeries.from_array(context, future_covariates)   # covariates: [B, F, T + 24]

out = model.predict(model_input, horizon=24, quantile_levels=[0.1, 0.5, 0.9])
```

`predict` infers `context_length` from where the forecast region starts. Pass it
explicitly when batching series of different widths. `forward` takes the same
`TimeSeries` and runs a single differentiable pass over it, with no rollout. That
is the entry point for fine-tuning.

**For efficient inference at scale, look at
[Retrocast](https://app.retrocast.com/).**

## Choose how to run `t0` models

**Model family:** [`t0-beta` (PyTorch/MLX)](https://huggingface.co/theforecastingcompany/t0-beta) · [`t0-alpha`](https://huggingface.co/theforecastingcompany/t0-alpha) · [Collection](https://huggingface.co/collections/theforecastingcompany/t0-model-family)

The `tfc-t0` package provides PyTorch and MLX runtimes as separate extras.
ONNX artifacts and our managed API cover other deployment targets:

| Use case | Install or open |
| --- | --- |
| Local inference with PyTorch | `pip install "tfc-t0[torch]>=0.5.0"` |
| Local inference on Apple silicon with MLX | `pip install "tfc-t0[mlx]>=0.6.0"` |
| Accelerator-oriented local and edge inference with ONNX FP16 | [`t0-alpha-onnx-fp16`](https://huggingface.co/theforecastingcompany/t0-alpha-onnx-fp16) |
| CPU and in-browser inference with ONNX INT8 | [`t0-alpha-onnx-int8`](https://huggingface.co/theforecastingcompany/t0-alpha-onnx-int8) |
| Managed inference without local weights | [The Forecasting Company API](https://docs.retrocast.com/documentation/t0-alpha) |

The ONNX artifacts are built from `t0-alpha`; there is no `t0-beta` ONNX
export yet.

You can also use `t0` on [Retrocast](https://app.retrocast.com/) to forecast on your own data and compare open-weight models.

## MLX runtime (Apple silicon)

`t0.mlx` is a first-party, inference-only runtime built on
[MLX](https://github.com/ml-explore/mlx). It needs Apple silicon, macOS 14 or
newer, and a native arm64 Python 3.10 or newer, and it does not install
PyTorch:

```bash
pip install "tfc-t0[mlx]>=0.6.0"
```

It loads the same checkpoints as the PyTorch runtime. MLX uses the Apple GPU
automatically; there is no device selection or `.to("mps")` step. `predict`
takes NumPy or MLX arrays:

```python
import numpy as np
from t0.mlx import T0Forecaster

model = T0Forecaster.from_pretrained("theforecastingcompany/t0-alpha").eval()

context = np.random.randn(4, 512).astype(np.float32)  # 4 series, 512 past timesteps
out = model.predict(context, horizon=64, quantile_levels=[0.1, 0.5, 0.9])
out.quantiles.shape  # (4, 64, 3)
out.median.shape     # (4, 64)
```

## 🏗️ Architecture

`t0` is a decoder-style patch transformer that alternates time and
covariate attention layers. It decodes multiple horizons in parallel — up
to 1024 timesteps in one forward pass — and falls back on autoregressive
rollout for longer horizons. Quantile levels between the trained ones are
interpolated, and levels beyond them extrapolated on exponential tails.

|                 | `t0-beta`                    | `t0-alpha`                |
| --------------- | ---------------------------- | ------------------------- |
| Parameters      | ~256M                        | ~102M                     |
| Layers          | 24                           | 24                        |
| Embedding dim   | 1024                         | 512                       |
| Feedforward dim | 2048                         | 2048                      |
| Attention heads | 8                            | 8                         |
| Patch size      | 32                           | 32                        |
| Quantile levels | 21, from 0.01 to 0.99        | 0.1, 0.25, 0.5, 0.75, 0.9 |
| `T0Config`      | `T0Config.large()`           | `T0Config.medium()`       |

### 🧬 Lineage

`t0` builds on ideas — and in places, code — from open-source forecasting
models. We gratefully acknowledge:

- **Toto** by Datadog ([repo](https://github.com/DataDog/toto)) &
  **Chronos-2** by Amazon
  ([repo](https://github.com/amazon-science/chronos-forecasting)) —
  factorizing attention in the time and variates dimension.
- **TiRex** by NXAI
  ([repo](https://github.com/NX-AI/tirex)) — contiguous patch masking.
- **rotary-embedding-torch** by Phil Wang
  ([repo](https://github.com/lucidrains/rotary-embedding-torch)) — rotary
  position embeddings, used by the PyTorch runtime and followed by the MLX
  one.

Code-level attributions are listed in [`NOTICE`](NOTICE), under their
respective open-source licenses.

## 📚 Citation

If our model is useful, please cite our [paper](https://arxiv.org/abs/2609.24559) and star our repo!

```bibtex
@article{meyer2026t0,
  title   = {$t_0$: A Time-Series Foundation Model for Forecasting with Context},
  author  = {Meyer, Lucas and Sole, Claudio and Xiang, Huikan and Li, Nicolas and Franceschino, Lucas and Quera-Bofarull, Arnau and Scholl, Maarten P. and Fainberg, Joachim and N{\'e}giar, Geoffrey},
  journal = {arXiv preprint arXiv:2609.24559},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.24559},
}
```

## ⚖️ License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
