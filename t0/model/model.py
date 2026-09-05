# Copyright 2026 The Forecasting Company
# The architecture is inspired by Datadog's Toto patched-transformer backbone
# (https://github.com/DataDog/toto).
# Copyright 2025 Datadog, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Transformer backbone and ``predict`` API for the open-weights t0-alpha model."""

import contextlib
import dataclasses
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from jaxtyping import Float, Int
from torch import Tensor

from t0.config import T0Config
from t0.data import TimeSeries
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
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        revision: str | None = None,
        force_download: bool = False,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        **model_kwargs: object,
    ) -> Self:
        """Load a checkpoint, preserving errors when its Hub config is inaccessible.

        For gated models, accept the model's access conditions and authenticate
        with ``hf auth login`` or pass ``token``. Local directories and cached
        checkpoints remain usable without network access.
        """
        if not Path(pretrained_model_name_or_path).is_dir():
            # The Hub mixin treats config HTTP errors as an optional missing file.
            # T0 requires its architecture config: check access first so a denied
            # download raises the original error rather than a constructor TypeError.
            hf_hub_download(
                str(pretrained_model_name_or_path),
                CONFIG_NAME,
                revision=revision,
                force_download=force_download,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        return super().from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            force_download=force_download,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **model_kwargs,
        )

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

    @torch.inference_mode()
    def predict(
        self,
        context: Float[Tensor, "batch time"]
        | Float[Tensor, "batch variates time"]
        | Float[np.ndarray, "batch time"]
        | Float[np.ndarray, "batch variates time"],
        horizon: int,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        future_covariates: Float[Tensor, "batch future_variates context_plus_horizon"]
        | Float[np.ndarray, "batch future_variates context_plus_horizon"]
        | None = None,
        mask: Int[Tensor, "*batch time"] | Int[np.ndarray, "*batch time"] | None = None,
        group_ids: Int[Tensor, " rows"] | Int[np.ndarray, " rows"] | None = None,
    ) -> Forecast:
        """Forecast ``horizon`` future timesteps for a batch of series.

        Args:
            context: Past observations — ``[T]`` / ``[B, T]`` (independent
                univariate series) or ``[B, V, T]`` (multivariate, jointly
                forecast). NaN marks a missing observation unless ``mask`` says
                otherwise.
            horizon: Number of future timesteps to forecast.
            quantiles: Quantile levels to return, sorted ascending in
                ``(0, 1)``; levels the model wasn't trained on are interpolated.
            future_covariates: Optional ``[B, F, T + horizon]`` covariates known
                over the context and horizon (e.g. calendar features);
                conditioned on but not forecast. NaN over the horizon is 0.
            mask: ``MaskType`` values shaped like ``context``: ``MISSING`` for an
                absent observation, ``PAD`` for a cell that only pads a shorter
                series out to the batch's width.
                Defaults to reading every NaN in ``context`` as a missing observation.
                Only all-``PAD`` patches are left unattended.
            group_ids: One id per row of the context — per row of a ``[B, T]``
                one, per sample-variate row of a ``[B, V, T]`` one — marking
                which rows are variates of the same series; rows sharing an id
                are forecast jointly. Defaults to one series per sample. Cannot
                be combined with ``future_covariates``.

        Returns:
            A forecast with quantiles shaped ``[B, horizon, Q]`` or
            ``[B, V, horizon, Q]`` — float32, finite, on the model's device.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if not quantiles:
            raise ValueError("quantiles must be non-empty")
        for q in quantiles:
            if not (0.0 < q < 1.0):
                raise ValueError(f"each quantile must be in (0, 1); got {q}")
        if list(quantiles) != sorted(set(quantiles)):
            raise ValueError(f"quantiles must be sorted ascending without duplicates; got {list(quantiles)}")

        context_t = torch.as_tensor(context)
        if context_t.ndim == 1:
            context_t = context_t.unsqueeze(0)
        if context_t.ndim not in (2, 3):
            raise ValueError(f"context must be [T], [B, T] or [B, V, T]; got shape {tuple(context_t.shape)}")
        device = next(self.parameters()).device
        context_t = context_t.to(device=device, dtype=torch.float32)

        future_t = None
        if future_covariates is not None:
            future_t = torch.as_tensor(future_covariates)
            expected_len = context_t.shape[-1] + horizon
            if future_t.ndim != 3 or future_t.shape[0] != context_t.shape[0] or future_t.shape[2] != expected_len:
                raise ValueError(
                    f"future_covariates must be [B={context_t.shape[0]}, F, T+horizon={expected_len}]; "
                    f"got shape {tuple(future_t.shape)}"
                )
            future_t = future_t.to(device=device, dtype=torch.float32)

        group_t = None
        if group_ids is not None:
            group_t = torch.as_tensor(group_ids).to(device=device)

        mask_t = None
        if mask is not None:
            mask_t = torch.as_tensor(mask)
            # Mirror the 1-D promotion applied to `context` so the shapes line up.
            if mask_t.ndim == 1:
                mask_t = mask_t.unsqueeze(0)
            mask_t = mask_t.to(device=device)

        model_input = TimeSeries.from_array(context_t, future_t, mask=mask_t, group_ids=group_t)
        # Autocast the forward when bf16/fp16 was requested: matmul/linear in
        # that dtype, softmax/norm in fp32.
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
                context_length=context_t.shape[-1],
            )
        if context_t.ndim == 3:
            predictions = predictions.unflatten(0, context_t.shape[:2])
        return Forecast(quantiles=_sanitize_predictions(predictions), quantile_levels=tuple(quantiles))


def _sanitize_predictions(predictions: Float[Tensor, "*batch quantiles"]) -> Float[Tensor, "*batch quantiles"]:
    """Cast to float32 and replace NaN/Inf with 0.0 (logged), so callers never see a poisoned tensor."""
    # Cast before nan_to_num: bf16 nan_to_num has surprising behavior with ±inf.
    predictions = predictions.float()
    non_finite_count = int((~predictions.isfinite()).sum().item())
    if non_finite_count > 0:
        logger.warning("replaced %d non-finite prediction values with 0.0", non_finite_count)
        predictions = predictions.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    return predictions
