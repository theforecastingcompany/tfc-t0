# Copyright 2026 The Forecasting Company
# The architecture is inspired by Datadog's Toto patched-transformer backbone
# (https://github.com/DataDog/toto).
# Copyright 2025 Datadog, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Transformer backbone for the open-weights t0-alpha model.

``T0Forecaster.forward`` runs one differentiable pass; ``T0Forecaster.predict`` wraps it
with scaling, rollout and inference mode.
"""

import contextlib
import dataclasses
import logging
import sys
from collections.abc import Sequence
from typing import overload

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Float, Int
from torch import Tensor

from t0.config import T0Config
from t0.data import TimeSeries, VariateType, time_series_from_array
from t0.mask import MaskBuilder
from t0.model.layers import PatchEncoder, Patcher, QuantileHead, ResidualBlock, Transformer
from t0.model.rollout import RolloutManager
from t0.quantile import interpolate_quantiles
from t0.scaler import CausalScaler

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

logger = logging.getLogger(__name__)

# Steps predicted per forward pass; longer horizons continue with the rollout.
DEFAULT_MAX_HORIZON = 1024


@dataclasses.dataclass
class Forecast:
    """Quantile forecast.

    ``quantiles`` is ``(B, horizon, Q)`` or ``(B, V, horizon, Q)``, last axis
    ordered like ``quantile_levels``.
    """

    quantiles: Float[Tensor, "batch horizon quantiles"] | Float[Tensor, "batch variates horizon quantiles"]
    quantile_levels: tuple[float, ...]

    @property
    def median(self) -> Float[Tensor, "batch horizon"] | Float[Tensor, "batch variates horizon"]:
        """The 0.5 quantile — exact when requested, otherwise interpolated from ``quantiles``."""
        if 0.5 in self.quantile_levels:
            return self.quantiles[..., self.quantile_levels.index(0.5)]
        logger.debug("0.5 not among quantile_levels %s — interpolating the median", self.quantile_levels)
        levels = torch.tensor(self.quantile_levels, dtype=torch.float32, device=self.quantiles.device)
        query = torch.tensor([0.5], dtype=torch.float32, device=self.quantiles.device)
        return interpolate_quantiles(query, levels, self.quantiles)[..., 0]


class T0Forecaster(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="tfc-t0",
    repo_url="https://github.com/theforecastingcompany/tfc-t0",
    pipeline_tag="time-series-forecasting",
    license="apache-2.0",
    tags=["time-series", "forecasting", "foundation-models", "pretrained-models", "safetensors"],
):
    """Open-weights t0-alpha forecasting backbone.

    Construct with explicit hyperparameters, or via ``from_config`` /
    ``from_pretrained``. ``embed_dim`` must be divisible by ``num_heads`` and
    ``group_every_n`` must divide ``num_layers``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_hidden_dim: int,
        patch_size: int,
        group_every_n: int,
        dropout: float,
        quantile_levels: Sequence[float],
        scaler_use_arcsinh: bool = True,
        # Forwarded by huggingface_hub 1.x's from_pretrained (renamed from
        # torch_dtype). bf16/fp16 keep fp32 weights and autocast the forward
        # in predict(); other dtypes run in fp32 with no autocast.
        dtype: torch.dtype | None = None,
        **_: object,
    ):
        super().__init__()
        # config.json is the single serialized source of truth; building it validates the quantile levels.
        self.config = T0Config(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            patch_size=patch_size,
            group_every_n=group_every_n,
            dropout=dropout,
            quantile_levels=tuple(quantile_levels),
            scaler_use_arcsinh=scaler_use_arcsinh,
        )

        self.patch_size = patch_size
        # Not in the serialized config — callers may tune it (keep a multiple of patch_size).
        self.max_horizon = DEFAULT_MAX_HORIZON
        self.patcher = Patcher(patch_size=patch_size)
        self.head = QuantileHead(quantile_levels=list(quantile_levels))
        self.mask_builder = MaskBuilder()
        # Per-time-step causal scaling (granularity 1): every time step is
        # standardized by its own running statistics — matching how the
        # published checkpoint was trained. Forecasts are rescaled with the
        # stats at each patch's last time step (see rescale_predictions).
        self.scaler = CausalScaler(patch_size=1, use_arcsinh=scaler_use_arcsinh)

        self.patch_encoder = PatchEncoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
            activation=nn.ReLU,
        )

        self.transformer = Transformer(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout,
            group_every_n=group_every_n,
            mask_builder=self.mask_builder,
        )

        self.decoder = ResidualBlock(
            input_size=embed_dim,
            hidden_size=embed_dim,
            output_size=patch_size * self.head.n_quantiles,
            activation=nn.ReLU,
        )

        # bf16/fp16 autocast the forward in predict() (weights stay fp32);
        # anything else runs in fp32.
        self._amp_dtype: torch.dtype | None = dtype if dtype in (torch.float16, torch.bfloat16) else None

    @classmethod
    def from_config(cls, config: T0Config) -> Self:
        """Build a fresh, randomly initialized model from a config."""
        return cls(**dataclasses.asdict(config))

    def forward(self, model_input: TimeSeries) -> Float[Tensor, "variates patches patch_size quantiles"]:
        """Predict per-patch quantiles for every patch position."""
        padded_input = self.patcher.pad(model_input)

        patched_values = self.patcher.patch(padded_input.variates)
        patched_mask = self.patcher.patch(padded_input.mask)
        patched_variate_type = self.patcher.patch(padded_input.variate_type)
        patched_group_ids = self.patcher.patch(padded_input.group_ids)

        embeddings = self.patch_encoder(patched_values, patched_mask, patched_variate_type)
        embeddings = self.transformer(embeddings, patched_group_ids, patched_variate_type, patched_mask)

        decoded = self.decoder(embeddings).unflatten(-1, (self.patch_size, self.head.n_quantiles))
        return self.head(decoded)

    @staticmethod
    def _validate_prediction_args(horizon: int, quantiles: Sequence[float]) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if not quantiles:
            raise ValueError("quantiles must be non-empty")
        for q in quantiles:
            if not (0.0 < q < 1.0):
                raise ValueError(f"each quantile must be in (0, 1); got {q}")
        if list(quantiles) != sorted(set(quantiles)):
            raise ValueError(f"quantiles must be sorted ascending without duplicates; got {list(quantiles)}")

    @overload
    def predict(
        self,
        model_input: TimeSeries,
        horizon: int,
        quantiles: Sequence[float] = ...,
        *,
        context_length: int | None = ...,
    ) -> Forecast: ...

    @overload
    def predict(
        self,
        model_input: Float[Tensor, "batch time"]
        | Float[Tensor, "batch variates time"]
        | Float[np.ndarray, "batch time"]
        | Float[np.ndarray, "batch variates time"],
        horizon: int,
        quantiles: Sequence[float] = ...,
        *,
        future_covariates: Float[Tensor, "batch future_variates context_plus_horizon"]
        | Float[np.ndarray, "batch future_variates context_plus_horizon"]
        | None = ...,
        mask: Int[Tensor, "*batch time"] | Int[np.ndarray, "*batch time"] | None = ...,
        group_ids: Int[Tensor, " rows"] | Int[np.ndarray, " rows"] | None = ...,
    ) -> Forecast: ...

    @torch.inference_mode()
    def predict(
        self,
        model_input,
        horizon: int,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        *,
        context_length: int | None = None,
        future_covariates=None,
        mask=None,
        group_ids=None,
    ) -> Forecast:
        """Forecast ``horizon`` future timesteps, continuing auto-regressively past ``max_horizon``.

        Args:
            model_input: A ``TimeSeries``, or past observations as ``[T]`` / ``[B, T]``
                (independent univariate series) or ``[B, V, T]`` (multivariate, jointly
                forecast). An array is converted with ``TimeSeries.from_array``. NaN marks a
                missing observation unless ``mask`` says otherwise.
            horizon: Number of future timesteps to forecast.
            quantiles: Quantile levels to return, sorted ascending in ``(0, 1)``; levels the
                model wasn't trained on are interpolated.
            context_length: Width of the right-aligned historical context. Defaults to the
                start of the forecast region.
            future_covariates: Array inputs only. ``[B, F, T + horizon]`` covariates known over
                the context and horizon (e.g. calendar features); conditioned on but not
                forecast. NaN over the horizon is 0.
            mask: Array inputs only. ``MaskType`` values shaped like the context: ``MISSING``
                for an absent observation, ``PAD`` for a cell that only pads a shorter series
                out to the batch's width. Defaults to reading every NaN as missing. Only
                all-``PAD`` patches are left unattended.
            group_ids: Array inputs only. One id per context row, marking which rows are
                variates of the same series; rows sharing an id are forecast jointly. Defaults
                to one series per sample. Cannot be combined with ``future_covariates``.

        Returns:
            Quantiles shaped ``[B, horizon, Q]`` or ``[B, V, horizon, Q]`` for array inputs,
            and ``[target_rows, horizon, Q]`` -- the input's row order -- for a ``TimeSeries``.
            Always float32, finite, on the model's device.

        Raises:
            ValueError: ``horizon`` or ``quantiles`` are out of range, ``context_length`` does
                not fit the input width, the input has no target row, or an array-only
                argument is passed alongside a ``TimeSeries``.
        """
        self._validate_prediction_args(horizon, quantiles)
        device = next(self.parameters()).device
        batch_shape = None

        if isinstance(model_input, TimeSeries):
            for name, value in (("future_covariates", future_covariates), ("mask", mask), ("group_ids", group_ids)):
                if value is not None:
                    raise ValueError(f"{name} applies to array inputs; build it into the TimeSeries instead")
        else:
            model_input, batch_shape = time_series_from_array(
                model_input, horizon, device, future_covariates, mask, group_ids
            )

        model_input = model_input.to(device)
        if context_length is None:
            context_length = self.patcher.context_end(model_input)
        if not 1 <= context_length <= model_input.seq_len:
            raise ValueError(
                f"context_length must be between 1 and the input width ({model_input.seq_len}), got {context_length}"
            )
        row_types = model_input.variate_type[:, -1]
        if not (row_types == VariateType.TARGET).any():
            raise ValueError("model_input must contain at least one target row")
        # An input either stops at the context or carries the forecast region. Known future
        # covariates must span that region, so they rule the context-only width out.
        widths = [context_length + horizon]
        if not (row_types == VariateType.FUTURE).any():
            widths.append(context_length)
        if model_input.seq_len not in widths:
            raise ValueError(
                f"model_input width must be {' or '.join(str(w) for w in sorted(widths))} "
                f"for context_length={context_length} and horizon={horizon}, got {model_input.seq_len}"
            )

        amp_ctx = (
            torch.autocast(device_type=device.type, dtype=self._amp_dtype)
            if self._amp_dtype is not None
            else contextlib.nullcontext()
        )
        with amp_ctx:
            predictions = RolloutManager(self).predict(
                model_input,
                prediction_length=horizon,
                query_quantile_levels=torch.tensor(list(quantiles), dtype=torch.float32, device=device),
                context_length=context_length,
            )
        predictions = _sanitize_predictions(predictions)
        if batch_shape is not None:
            predictions = predictions.unflatten(0, batch_shape)
        return Forecast(quantiles=predictions, quantile_levels=tuple(quantiles))


def _sanitize_predictions(predictions: Float[Tensor, "*batch quantiles"]) -> Float[Tensor, "*batch quantiles"]:
    """Cast to float32 and replace NaN/Inf with 0.0 (logged), so callers never see a poisoned tensor."""
    # Cast before nan_to_num: bf16 nan_to_num has surprising behavior with ±inf.
    predictions = predictions.float()
    non_finite_count = int((~predictions.isfinite()).sum().item())
    if non_finite_count > 0:
        logger.warning("replaced %d non-finite prediction values with 0.0", non_finite_count)
        predictions = predictions.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    return predictions
