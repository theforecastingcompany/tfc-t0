<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://www.theforecastingcompany.com/logo/logo_horizontal_dark.png" />
    <img src="https://www.theforecastingcompany.com/logo/logo_horizontal_light.png" alt="The Forecasting Company" width="280" />
  </picture>
</p>

# `t0`

<p align="center">
  <a href="https://arxiv.org/abs/2609.24559"><img src="https://img.shields.io/badge/arXiv-2609.24559-b31b1b.svg" alt="arXiv" /></a>
  <a href="https://pypi.org/project/tfc-t0/"><img src="https://img.shields.io/pypi/v/tfc-t0" alt="PyPI" /></a>
  <a href="https://pypi.org/project/tfc-t0/"><img src="https://img.shields.io/pypi/pyversions/tfc-t0" alt="Python versions" /></a>
  <a href="https://github.com/theforecastingcompany/tfc-t0/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/tfc-t0" alt="License" /></a>
  <a href="https://colab.research.google.com/github/theforecastingcompany/tfc-t0/blob/main/notebooks/01_inference_quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open inference quickstart in Colab" /></a>
</p>

Open-weights time-series forecasting foundation model from [The Forecasting Company](https://theforecastingcompany.com/).
`t0` is a transformer-based model that
produces probabilistic multi-horizon forecasts and natively operates on
multiple covariates. `t0-beta` is the current iteration: 256M parameters,
21 natively predicted quantile levels, and an incremental improvement over
`t0-alpha`.

You can use `t0` on [Retrocast](https://app.retrocast.com/), our platform for forecasting on your own data. You can also compare forecast across different open-weight models.

**Model family:** [`t0-beta` (PyTorch/MLX)](https://huggingface.co/theforecastingcompany/t0-beta) · [`t0-alpha`](https://huggingface.co/theforecastingcompany/t0-alpha) · [Collection](https://huggingface.co/collections/theforecastingcompany/t0-alpha-model-family-6a99be18a9e3ab245fda8501)

## Choose how to run `t0-beta`

The `tfc-t0` package contains the first-party PyTorch and MLX runtimes. Each
runtime is an extra, so an installation carries only the tensor framework it
uses. A bare `pip install tfc-t0` installs neither runtime: choose one with its
extra. ONNX artifacts and our managed API cover other deployment targets:

| Use case | Install or open |
| --- | --- |
| Local inference with PyTorch | `pip install "tfc-t0[torch]>=0.5.0"` |
| Local inference on Apple silicon with MLX | `pip install "tfc-t0[mlx]>=0.6.0"` |
| Accelerator-oriented local and edge inference with ONNX FP16 | [`t0-alpha-onnx-fp16`](https://huggingface.co/theforecastingcompany/t0-alpha-onnx-fp16) |
| CPU and in-browser inference with ONNX INT8 | [`t0-alpha-onnx-int8`](https://huggingface.co/theforecastingcompany/t0-alpha-onnx-int8) |
| Managed inference without local weights | [The Forecasting Company API](https://docs.retrocast.com/documentation/t0-alpha) |

The ONNX artifacts are built from `t0-alpha`; there is no `t0-beta` ONNX
export yet.

The [MLX runtime](#mlx-runtime-apple-silicon) is inference-only, has a closely
matched `T0Forecaster.predict()` API, loads the same safetensors directly, and
does not install PyTorch.

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

For a complete worked example, the
[European day-ahead electricity prices notebook](https://github.com/theforecastingcompany/tfc-t0/blob/main/notebooks/03_electricity_day_ahead_prices.ipynb)
forecasts hourly prices in four bidding zones with load, renewable, holiday
and weather covariates, and backtests the result.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theforecastingcompany/tfc-t0/blob/main/notebooks/03_electricity_day_ahead_prices.ipynb)

The [Quickstart](#-quickstart) below shows the API for both a plain
univariate forecast and a multivariate forecast that conditions on
historical and known-future covariates.

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

`t0.mlx` exports `T0Forecaster`, `Forecast`, `T0Config`, `TimeSeries`,
`MaskType`, `VariateType` and `batch_series`, with the same meaning as above:
both runtimes share one implementation of everything but the network layers,
so `predict` takes the same arguments, including a `TimeSeries`. It covers
univariate and multivariate inputs, missing values, explicit groups,
known-future covariates (`future_covariates=`), quantile interpolation with
exponential tails, and autoregressive rollout past the native 1024-step pass. Compilation is opt-in
and pays off for a shape you call repeatedly:

```python
model.compile()
out = model.predict(context, horizon=64)
```

On an Apple M1 Pro, compiled MLX was 3.37–3.59x faster than PyTorch MPS and
5.23–6.86x faster than PyTorch CPU across three representative `t0-alpha`
workloads, excluding model loading and the compiled path's first call. See
[`BENCHMARKS.md`](https://github.com/theforecastingcompany/tfc-t0/blob/main/BENCHMARKS.md) for the method and raw timings, and
[`PARITY.md`](https://github.com/theforecastingcompany/tfc-t0/blob/main/PARITY.md) for the checkpoint-backed numerical parity record.

The MLX parameter tree has the same tensor names, shapes and dtypes as the
checkpoint, so no conversion or separate MLX checkpoint is needed;
`tools/validate_mlx_checkpoint.py` checks that contract for a local
checkpoint. The runtime's tests live in `tests/mlx/`:

```bash
uv sync
uv run pytest tests/mlx
T0_MLX_CHECKPOINT=/path/to/t0-alpha uv run pytest tests/mlx  # + checkpoint-backed parity
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

## 🧰 Public API

`from t0 import ...` is the PyTorch runtime (also importable as `t0.torch`);
`t0.mlx` mirrors the names it shares, as described [above](#mlx-runtime-apple-silicon).

- `T0Forecaster` — `nn.Module` with `from_pretrained` /
  `save_pretrained` (via `huggingface_hub.PyTorchModelHubMixin`). It has two
  forecasting entry points. `forward(model_input)` runs a single differentiable
  pass with no rollout. `predict(model_input, horizon, quantile_levels, ...)` is
  inference-only and rolls out autoregressively past `max_horizon`.
- `Forecast` — the object returned by the model.
- `T0Config` — the configuration of the model. `T0Config.large()` is
  `t0-beta`; `T0Config.medium()` is `t0-alpha`.
- `MaskType` — the reason a time step is masked out: `PAD` (a cell that
  only widens a shorter series out to the batch's width) or `MISSING` (an
  absent observation).
- `batch_series` — utility to batch time series of potentially different
  lengths.
- `TimeSeries.from_array` / `TimeSeries.batch` — build the model's native
  input, including known-future covariates and an explicit forecast
  `horizon`. `predict` accepts either a `TimeSeries` or a raw context array.

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
