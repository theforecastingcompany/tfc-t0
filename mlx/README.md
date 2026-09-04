<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://www.theforecastingcompany.com/logo/logo_horizontal_dark.png" />
    <img src="https://www.theforecastingcompany.com/logo/logo_horizontal_light.png" alt="The Forecasting Company" width="280" />
  </picture>
</p>

# `t0` for MLX

<p align="center">
  <a href="https://pypi.org/project/tfc-t0-mlx/"><img src="https://img.shields.io/pypi/v/tfc-t0-mlx" alt="PyPI" /></a>
  <a href="https://pypi.org/project/tfc-t0-mlx/"><img src="https://img.shields.io/pypi/pyversions/tfc-t0-mlx" alt="Python versions" /></a>
  <a href="https://github.com/theforecastingcompany/tfc-t0/blob/main/mlx/LICENSE"><img src="https://img.shields.io/pypi/l/tfc-t0-mlx" alt="License" /></a>
</p>

Open-weights time-series forecasting foundation model from
[The Forecasting Company](https://theforecastingcompany.com/). `t0` is a
transformer-based model that produces probabilistic multi-horizon forecasts
and natively operates on multiple covariates. `t0-alpha` is our first iteration
of the model.

`tfc-t0-mlx` is the first-party inference runtime optimized for Apple silicon.
It loads the same original `t0-alpha` safetensors checkpoint as `tfc-t0` and
does not install PyTorch.

The MLX and PyTorch runtimes are developed together in this repository while
remaining separate PyPI distributions with independent dependency trees.

You can use `t0` on [Retrocast](https://app.retrocast.com/), our platform for
forecasting on your own data. You can also compare forecasts across different
open-weight models.

## Choose how to run `t0-alpha`

| Use case | Install or open |
| --- | --- |
| Local inference on Apple silicon with MLX | `pip install tfc-t0-mlx` |
| Local inference with PyTorch | [`pip install tfc-t0`](https://pypi.org/project/tfc-t0/) |
| Managed inference without local weights | [The Forecasting Company API](https://docs.retrocast.com/documentation/t0-alpha) |

Both local runtimes load the same model repository and offer a closely matched
`T0Forecaster.predict()` API. MLX compilation is available as an opt-in
optimization for repeated input shapes.

![t0 forecasting French national electricity demand in Retrocast](https://raw.githubusercontent.com/theforecastingcompany/tfc-t0/main/assets/enedis_with_holidays.webp)

_`t0` forecasting French national electricity demand in Retrocast. Data:
[Enedis open data](https://data.enedis.fr/)._

## 📈 Forecasting with covariates

`t0` leverages covariate information, in the past and future when available,
to improve its forecast.

| Without covariates | With covariates |
| --- | --- |
| ![t0 forecast without covariates](https://raw.githubusercontent.com/theforecastingcompany/tfc-t0/main/assets/medicam_without_cov.webp) | ![t0 forecast with covariates](https://raw.githubusercontent.com/theforecastingcompany/tfc-t0/main/assets/medicam_with_cov.webp) |

_Data: [Medic'AM](https://www.assurance-maladie.ameli.fr/etudes-et-donnees/medicaments-classe-atc-medicam),
monthly drug reimbursements from the French national health insurance._

The [Quickstart](#-quickstart) below shows the API for both a plain univariate
forecast and a multivariate forecast that conditions on historical and
known-future covariates.

## 🚀 Quickstart

`tfc-t0-mlx` requires Apple silicon, macOS 14 or newer, and a native arm64
Python 3.10 or newer. MLX uses the Apple GPU automatically; there is no
PyTorch-style device selection or `.to("mps")` step.

```bash
pip install tfc-t0-mlx
```

The model repository is gated. Accept its terms on
[Hugging Face](https://huggingface.co/theforecastingcompany/t0-alpha) and run
`hf auth login` before the first download.

The simplest path is a univariate forecast through `predict`:

```python
import numpy as np
from t0_mlx import T0Forecaster

model = T0Forecaster.from_pretrained("theforecastingcompany/t0-alpha").eval()

context = np.random.randn(4, 512).astype(np.float32)  # 4 series, 512 past timesteps
out = model.predict(context, horizon=64, quantiles=[0.1, 0.5, 0.9])
out.quantiles.shape  # (4, 64, 3)
out.median.shape     # (4, 64)
```

`predict` accepts NumPy and MLX arrays. One-dimensional contexts are
automatically promoted to a single-row batch. NaN in the context is read as a
missing observation; to say that some cells are padding instead, pass a `mask`
— see [batched inference](#batched-inference).

For a shape that will be called repeatedly, compilation is opt-in:

```python
model.compile()
out = model.predict(context, horizon=64)
```

### Forecasting with covariates

Anything you know over the **past** goes in `context` — alongside the target,
extra variates attend to it and are forecast together. Anything you know over
the **future** (calendar features, planned promotions, weather forecasts) goes
in `future_covariates`, shaped `[B, F, context + horizon]`; the model conditions
on it but does not forecast it.

```python
import numpy as np
from t0_mlx import T0Forecaster

model = T0Forecaster.from_pretrained("theforecastingcompany/t0-alpha").eval()

context = np.random.randn(2, 512).astype(np.float32)
future_covariates = np.random.randn(2, 3, 512 + 64).astype(np.float32)

out = model.predict(
    context,
    horizon=64,
    quantiles=[0.1, 0.5, 0.9],
    future_covariates=future_covariates,
)
out.quantiles.shape  # (2, 64, 3)
out.median.shape     # (2, 64)
```

### Batched inference

```python
import numpy as np
from t0_mlx import T0Forecaster, batch_series

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
out.quantiles.shape  # (4, 24, 3)
out.median[0]        # the 24-step median forecast for `daily`
```

**For efficient inference at scale, look at
[Retrocast](https://app.retrocast.com/).**

## 🏗️ Architecture

`t0` is a decoder-style patch transformer that alternates time and covariate
attention layers. It predicts 5 quantiles (0.1, 0.25, 0.5, 0.75, 0.9), decoding
multiple horizons in parallel — up to 1024 timesteps in one forward pass — and
falling back on autoregressive rollout for longer horizons.

|                 |                            |
| --------------- | -------------------------- |
| Parameters      | ~102M                      |
| Layers          | 24                         |
| Embedding dim   | 512                        |
| Feedforward dim | 2048                       |
| Attention heads | 8                          |
| Patch size      | 32                         |
| Quantile levels | 0.1, 0.25, 0.5, 0.75, 0.9 |

### Performance

On an Apple M1 Pro, compiled MLX was 2.62–2.91x faster than PyTorch MPS and
4.83–5.38x faster than PyTorch CPU across three representative workloads. MLX
eager was 1.97–2.15x faster than PyTorch MPS. These measurements exclude model
loading and the compiled path's shape-dependent first call.

See the [complete benchmark](BENCHMARKS.md) for the reproducible script,
environment, methodology, raw timings, and compilation caveats.

### 🧬 Lineage

`t0` builds on ideas — and in places, code — from open-source forecasting
models. We gratefully acknowledge:

- **Toto** by Datadog ([repo](https://github.com/DataDog/toto)) and
  **Chronos-2** by Amazon
  ([repo](https://github.com/amazon-science/chronos-forecasting)) — factorizing
  attention in the time and variates dimensions.
- **TiRex** by NXAI
  ([repo](https://github.com/NX-AI/tirex)) — contiguous patch masking.
- **rotary-embedding-torch** by Phil Wang
  ([repo](https://github.com/lucidrains/rotary-embedding-torch)) — rotary
  position embedding behavior.

Code-level attributions are listed in [`NOTICE`](NOTICE), under their
respective open-source licenses. This MLX implementation is based on the
first-party PyTorch architecture and checkpoint.

## 🧰 Public API

The top-level API mirrors `tfc-t0`:

- `T0Forecaster` — MLX `nn.Module` with `from_pretrained`, `save_pretrained`,
  opt-in `compile` / `uncompile`, and the user-facing `predict(context, horizon,
  quantiles, future_covariates, mask, group_ids)`.
- `Forecast` — the object returned by the model.
- `T0Config` — the configuration of the model; `T0Config.medium()` is the
  published one.
- `MaskType` — the reason a time step is masked out, including `PAD` and
  `MISSING`.
- `batch_series` — utility to batch time series of potentially different
  lengths.

The runtime supports univariate and multivariate inputs, missing values,
explicit groups, known-future covariates, requested-quantile interpolation,
and arbitrary positive horizons. Forecasts beyond the native 1024-step pass
continue autoregressively, matching `tfc-t0`.

## Development

From the repository root:

```bash
cd mlx
uv sync
uv run pytest
uv run ruff check .
```

To run checkpoint-backed parity tests, point them at a local directory holding
the original `config.json` and `model.safetensors`. The checked release record
is in [`PARITY.md`](PARITY.md):

```bash
T0_MLX_CHECKPOINT=/path/to/t0-alpha uv run --extra parity pytest
```

The MLX parameter tree has the same tensor names, shapes, dtypes, and
linear-layer orientation as the original checkpoint, so no numerical weight
conversion or separate MLX checkpoint is required.
`tools/validate_checkpoint.py` validates this contract without copying or
rewriting the weights.

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
