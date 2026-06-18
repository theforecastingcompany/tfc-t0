<p align="center">
  <img src="https://www.theforecastingcompany.com/logo/logo_horizontal_pride_universal.png" alt="The Forecasting Company" width="280" />
</p>

# `t0`

Open-weights time-series forecasting foundation model from [The Forecasting Company](https://theforecastingcompany.com/).
`t0` is a transformer-based model that
produces probabilistic multi-horizon forecasts and natively operates on
multiple covariates. `t0-alpha` is our first iteration of the model.

You can use `t0` on [Retrocast](https://app.retrocast.com/), our platform for forecasting on your own data. You can also compare forecast across different open-weight models.

![t0 forecasting French national electricity demand in Retrocast](https://huggingface.co/theforecastingcompany/t0-alpha/resolve/main/assets/enedis_with_holidays.png)

_`t0` forecasting French national electricity demand in Retrocast. Data:
[Enedis open data](https://data.enedis.fr/)._

## 📈 Forecasting with covariates

`t0` leverages covariate information, in the past and future when
available, to improve its forecast.

| Without covariates                                                                                                                   | With covariates                                                                                                                |
| ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| ![t0 forecast without covariates](https://huggingface.co/theforecastingcompany/t0-alpha/resolve/main/assets/medicam_without_cov.png) | ![t0 forecast with covariates](https://huggingface.co/theforecastingcompany/t0-alpha/resolve/main/assets/medicam_with_cov.png) |

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
single-row batch. NaN values in the context are treated as missing
observations.

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
  user-facing `predict(context, horizon, quantiles, future_covariates)`.
- `T0Config` — frozen dataclass; `T0Config.medium()` is the published
  configuration.

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
