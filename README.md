<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://www.theforecastingcompany.com/logo/logo_horizontal_dark.png" />
    <img src="https://www.theforecastingcompany.com/logo/logo_horizontal_light.png" alt="The Forecasting Company" width="280" />
  </picture>
</p>

# `t0`

<p align="center">
  <a href="https://pypi.org/project/tfc-t0/"><img src="https://img.shields.io/pypi/v/tfc-t0" alt="PyPI" /></a>
  <a href="https://pypi.org/project/tfc-t0/"><img src="https://img.shields.io/pypi/pyversions/tfc-t0" alt="Python versions" /></a>
  <a href="https://github.com/theforecastingcompany/tfc-t0/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/tfc-t0" alt="License" /></a>
</p>

Open-weights time-series forecasting foundation model from [The Forecasting Company](https://theforecastingcompany.com/).
`t0` is a transformer-based model that
produces probabilistic multi-horizon forecasts and natively operates on
multiple covariates. `t0-alpha` is our first iteration of the model.

You can use `t0` on [Retrocast](https://app.retrocast.com/), our platform for forecasting on your own data. You can also compare forecast across different open-weight models.

**Model family:** [`t0-alpha` (PyTorch/MLX)](https://huggingface.co/theforecastingcompany/t0-alpha) · [ONNX FP16](https://huggingface.co/theforecastingcompany/t0-alpha-onnx-fp16) · [ONNX INT8](https://huggingface.co/theforecastingcompany/t0-alpha-onnx-int8) · [Collection](https://huggingface.co/collections/theforecastingcompany/t0-alpha-model-family-6a99be18a9e3ab245fda8501)

## Choose how to run `t0-alpha`

This package is the first-party PyTorch runtime. First-party MLX and ONNX
options, plus our managed API, are available for other deployment targets:

| Use case | Install or open |
| --- | --- |
| Local inference with PyTorch | `pip install tfc-t0` |
| Local inference on Apple silicon with MLX | [`pip install tfc-t0-mlx`](https://pypi.org/project/tfc-t0-mlx/) |
| Accelerator-oriented local and edge inference with ONNX FP16 | [`t0-alpha-onnx-fp16`](https://huggingface.co/theforecastingcompany/t0-alpha-onnx-fp16) |
| CPU and in-browser inference with ONNX INT8 | [`t0-alpha-onnx-int8`](https://huggingface.co/theforecastingcompany/t0-alpha-onnx-int8) |
| Managed inference without local weights | [The Forecasting Company API](https://docs.retrocast.com/documentation/t0-alpha) |

The MLX runtime is inference-only, has a similar `T0Forecaster.predict()` API,
loads this model's safetensors directly and does not install PyTorch.

![t0 forecasting French national electricity demand in Retrocast](https://raw.githubusercontent.com/theforecastingcompany/tfc-t0/main/assets/enedis_with_holidays.webp)

_`t0` forecasting French national electricity demand in Retrocast. Data:
[Enedis open data](https://data.enedis.fr/)._

## 📈 Forecasting with covariates

`t0` leverages covariate information, in the past and future when
available, to improve its forecast.

| Without covariates                                                                                                                   | With covariates                                                                                                                |
| ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| ![t0 forecast without covariates](https://raw.githubusercontent.com/theforecastingcompany/tfc-t0/main/assets/medicam_without_cov.webp) | ![t0 forecast with covariates](https://raw.githubusercontent.com/theforecastingcompany/tfc-t0/main/assets/medicam_with_cov.webp) |

_Data: [Medic'AM](https://www.assurance-maladie.ameli.fr/etudes-et-donnees/medicaments-classe-atc-medicam),
monthly drug reimbursements from the French national health insurance._

The [Quickstart](#-quickstart) below shows the API for both a plain
univariate forecast and a multivariate forecast that conditions on
historical and known-future covariates.

## 🚀 Quickstart

```bash
pip install tfc-t0
```

The simplest path is a univariate forecast through `predict`:

```python
import torch
from t0 import T0Forecaster

model = T0Forecaster.from_pretrained("theforecastingcompany/t0-alpha").eval()

context = torch.randn(4, 512)  # 4 series, 512 past timesteps
out = model.predict(context, horizon=64, quantiles=[0.1, 0.5, 0.9])
out.quantiles  # (4, 64, 3)
out.median     # (4, 64)
```

`predict` accepts `numpy` arrays. 1-D contexts are auto-promoted to a
single-row batch. NaN in the context is read as a missing observation; to
say that some cells are padding instead, pass a `mask` — see
[batched inference](#batched-inference).

### Forecasting with covariates

Anything you know over the **past** goes in `context` — alongside the
target, extra variates attend to it and are forecast together. Anything
you know over the **future** (calendar features, planned promotions,
weather forecasts) goes in `future_covariates`, shaped
`[B, F, context + horizon]`; the model conditions on it but does not
forecast it.

```python
import torch
from t0 import T0Forecaster

model = T0Forecaster.from_pretrained("theforecastingcompany/t0-alpha").eval()

context = torch.randn(2, 512)                    # 2 series, 512 past timesteps
future_covariates = torch.randn(2, 3, 512 + 64)  # 3 covariates known over context + horizon

out = model.predict(
    context,
    horizon=64,
    quantiles=[0.1, 0.5, 0.9],
    future_covariates=future_covariates,
)
out.quantiles  # (2, 64, 3)
out.median     # (2, 64)
```

### Batched inference

```python
import numpy as np
from t0 import T0Forecaster, batch_series

model = T0Forecaster.from_pretrained("theforecastingcompany/t0-alpha").eval()

daily = np.random.randn(180)    # one series, 180 past timesteps
store = np.random.randn(2, 96)  # one series of 2 variates, 96 past timesteps
hourly = np.random.randn(1024)  # one series, 1024 past timesteps

context, mask, group_ids = batch_series([daily, store, hourly])
context.shape  # (4, 1024) — variates stacked, right-aligned to the longest
group_ids      # [0, 1, 1, 2] — `store`'s two variates are forecast jointly

out = model.predict(
    context,
    horizon=24,
    quantiles=[0.1, 0.5, 0.9],
    mask=mask,
    group_ids=group_ids,
)
out.quantiles  # (4, 24, 3)
out.median[0]  # the 24-step median forecast for `daily`
```

**For efficient inference at scale, look at
[Retrocast](https://app.retrocast.com/).**

## 🏗️ Architecture

`t0` is a decoder-style patch transformer that alternates time and
covariate attention layers. It predicts 5 quantiles (0.1, 0.25, 0.5,
0.75, 0.9), decoding multiple horizons in parallel — up to 1024
timesteps in one forward pass — and falling back on autoregressive
rollout for longer horizons.

|                 |                           |
| --------------- | ------------------------- |
| Parameters      | ~102M                     |
| Layers          | 24                        |
| Embedding dim   | 512                       |
| Feedforward dim | 2048                      |
| Attention heads | 8                         |
| Patch size      | 32                        |
| Quantile levels | 0.1, 0.25, 0.5, 0.75, 0.9 |

### 🧬 Lineage

`t0` builds on ideas — and in places, code — from open-source forecasting
models. We gratefully acknowledge:

- **Toto** by Datadog ([repo](https://github.com/DataDog/toto)) &
  **Chronos-2** by Amazon
  ([repo](https://github.com/amazon-science/chronos-forecasting)) —
  factorizing attention in the time and variates dimension.
- **TiRex** by NXAI
  ([repo](https://github.com/NX-AI/tirex)) — contiguous patch masking.

Code-level attributions are listed in [`NOTICE`](NOTICE), all under
Apache-2.0.

## 🧰 Public API

- `T0Forecaster` — `nn.Module` with `from_pretrained` /
  `save_pretrained` (via `huggingface_hub.PyTorchModelHubMixin`) and the
  user-facing `predict(context, horizon, quantiles, future_covariates,
  mask, group_ids)`.
- `Forecast` — the object returned by the model.
- `T0Config` — the configuration of the model; `T0Config.medium()` is the
  published one.
- `MaskType` — the reason a time step is masked out: `PAD` (a cell that
  only widens a shorter series out to the batch's width) or `MISSING` (an
  absent observation).
- `batch_series` — utility to batch time series of potentially different
  lengths.

## 📚 Citation

If our model is useful, please use the following citation and star our repo!

```bibtex
@misc{tfc-t0,
  title  = {t0: A time-series forecasting foundation model},
  author = {The Forecasting Company},
  year   = {2026},
  url    = {https://huggingface.co/theforecastingcompany/t0-alpha},
}
```

## ⚖️ License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
