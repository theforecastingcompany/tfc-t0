# Copyright 2026 The Forecasting Company
# The architecture follows the first-party t0 PyTorch implementation, which is
# inspired by Datadog's Toto patched-transformer backbone
# (https://github.com/DataDog/toto). The autoregressive quantile rollout follows
# Chronos-2's inference approach
# (https://github.com/amazon-science/chronos-forecasting).
# Copyright 2025 Datadog, Inc.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLX-native t0-alpha model and inference API."""

import dataclasses
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download

from t0_mlx.config import T0Config
from t0_mlx.data import MaskType, TimeSeries, VariateType
from t0_mlx.layers import PatchEncoder, Patcher, QuantileHead, ResidualBlock, Transformer
from t0_mlx.quantile import interpolate_quantiles, reduce_rollout_quantiles, validate_quantiles
from t0_mlx.scaler import CausalScaler

logger = logging.getLogger(__name__)

DEFAULT_MAX_HORIZON = 1024


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _sanitize_predictions(predictions: mx.array) -> mx.array:
    """Cast to float32 and replace NaN/Inf with 0.0 while making the failure visible."""
    predictions = predictions.astype(mx.float32)
    non_finite_count = mx.sum(~mx.isfinite(predictions))
    mx.eval(non_finite_count)
    count = int(np.asarray(non_finite_count).item())
    if count > 0:
        logger.warning("replaced %d non-finite prediction values with 0.0", count)
        predictions = mx.where(mx.isfinite(predictions), predictions, mx.array(0.0, dtype=mx.float32))
    return predictions


@dataclasses.dataclass
class Forecast:
    """Quantile forecast.

    ``quantiles`` is ``(B, horizon, Q)`` or ``(B, V, horizon, Q)``, with
    the last axis ordered like ``quantile_levels``.
    """

    quantiles: mx.array
    quantile_levels: tuple[float, ...]

    @property
    def median(self) -> mx.array:
        """The 0.5 quantile, interpolated from ``quantiles`` when needed."""
        if 0.5 in self.quantile_levels:
            return self.quantiles[..., self.quantile_levels.index(0.5)]
        return interpolate_quantiles((0.5,), self.quantile_levels, self.quantiles)[..., 0]


class T0Forecaster(nn.Module):
    """Open-weights t0-alpha forecasting backbone for MLX inference.

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
        **_: Any,
    ):
        super().__init__()
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
        self.max_horizon = DEFAULT_MAX_HORIZON
        self.patcher = Patcher(patch_size)
        self.head = QuantileHead(quantile_levels)
        # The PyTorch T0Forecaster deliberately constructs its scaler with
        # patch_size=1: the published checkpoint was trained with per-time-step
        # running statistics. rescale_predictions selects the statistic at each
        # model patch's right edge.
        self.scaler = CausalScaler(use_arcsinh=scaler_use_arcsinh)
        self.patch_encoder = PatchEncoder(embed_dim, patch_size)
        self.transformer = Transformer(
            num_layers,
            embed_dim,
            num_heads,
            mlp_hidden_dim,
            dropout,
            group_every_n,
        )
        self.decoder = ResidualBlock(embed_dim, embed_dim, patch_size * self.head.n_quantiles)
        self._compiled_single_pass = None

    @classmethod
    def from_config(cls, config: T0Config) -> "T0Forecaster":
        """Construct an uninitialized model from architecture configuration."""
        return cls(**config.to_dict())

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path,
        *,
        revision: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
    ) -> "T0Forecaster":
        """Load `config.json` and the original safetensors checkpoint.

        The published PyTorch tensor layouts are already compatible with MLX,
        so loading is strict and does not transpose or numerically convert any
        model weight.
        """
        candidate = Path(model_id_or_path).expanduser()
        if candidate.is_dir():
            model_path = candidate
        else:
            model_path = Path(
                snapshot_download(
                    repo_id=str(model_id_or_path),
                    revision=revision,
                    token=token,
                    local_files_only=local_files_only,
                    allow_patterns=["config.json", "model.safetensors"],
                )
            )
        config_path = model_path / "config.json"
        weights_path = model_path / "model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(f"{model_path} must contain config.json and model.safetensors")

        model = cls.from_config(T0Config.from_json(config_path))
        model.load_weights(str(weights_path), strict=True)
        model.eval()
        mx.eval(model.parameters())
        return model

    def save_pretrained(self, directory: str | Path) -> None:
        """Write a strict-loadable `config.json` and `model.safetensors`."""
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "config.json").open("w", encoding="utf-8") as config_file:
            json.dump(self.config.to_dict(), config_file, indent=2)
            config_file.write("\n")
        self.save_weights(str(destination / "model.safetensors"))

    def __call__(self, model_input: TimeSeries) -> mx.array:
        """Return per-patch values on the model's native quantile grid."""
        padded = self.patcher.pad(model_input)
        values = self.patcher.patch(padded.variates)
        mask = self.patcher.patch(padded.mask)
        variate_type = self.patcher.patch(padded.variate_type)
        group_ids = self.patcher.patch(padded.group_ids)

        embeddings = self.patch_encoder(values, mask, variate_type)
        embeddings = self.transformer(embeddings, group_ids, variate_type, mask)
        decoded = self.decoder(embeddings)
        decoded = decoded.reshape(*decoded.shape[:-1], self.patch_size, self.head.n_quantiles)
        return self.head(decoded)

    def compile(self, *, shapeless: bool = False) -> "T0Forecaster":
        """Compile the scaling, model, and rescaling graph for repeated inference."""
        self._compiled_single_pass = mx.compile(
            self._single_pass_arrays,
            inputs=self.state,
            shapeless=shapeless,
        )
        return self

    def uncompile(self) -> "T0Forecaster":
        """Return to eager MLX execution."""
        self._compiled_single_pass = None
        return self

    def predict(
        self,
        context: mx.array | np.ndarray | list[Any] | tuple[Any, ...],
        horizon: int,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        future_covariates: mx.array | np.ndarray | None = None,
        mask: mx.array | np.ndarray | None = None,
        group_ids: mx.array | np.ndarray | None = None,
    ) -> Forecast:
        """Forecast ``horizon`` future timesteps for a batch of series.

        Args:
            context: Past observations — ``[T]`` / ``[B, T]`` (independent
                univariate series) or ``[B, V, T]`` (multivariate, jointly
                forecast). NaN marks a missing observation unless ``mask`` says
                otherwise.
            horizon: Number of future timesteps to forecast.
            quantiles: Quantile levels to return, sorted ascending in
                ``(0, 1)``; levels the model was not trained on are interpolated.
            future_covariates: Optional ``[B, F, T + horizon]`` covariates known
                over the context and horizon (for example calendar features);
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
            A float32, finite forecast with quantiles shaped
            ``[B, horizon, Q]`` or ``[B, V, horizon, Q]``.

        Up to ``max_horizon`` timesteps are decoded in one pass; longer
        horizons continue autoregressively.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if self.max_horizon < self.patch_size or self.max_horizon % self.patch_size != 0:
            raise ValueError(
                f"max_horizon must be a positive multiple of patch_size ({self.patch_size}), got {self.max_horizon}"
            )
        requested_quantiles = validate_quantiles(quantiles)

        context_array = mx.array(context, dtype=mx.float32)
        mask_array = None if mask is None else mx.array(mask)
        group_array = None if group_ids is None else mx.array(group_ids)
        future_array = None if future_covariates is None else mx.array(future_covariates, dtype=mx.float32)
        model_input = TimeSeries.from_context(
            context_array,
            mask_array,
            group_array,
            future_array,
            horizon,
        )
        target_count = context_array.size // context_array.shape[-1]

        forecast_width = _round_up(horizon, self.patch_size)
        context_length = context_array.shape[-1]
        context_width = _round_up(context_length, self.patch_size)
        buffer = self._prepare_single_pass_buffer(model_input, forecast_width, context_length)

        step_horizon = min(forecast_width, self.max_horizon)
        window = buffer.time_slice(0, context_width + step_horizon)
        native = self._predict_step(window, step_horizon)[:target_count]
        prediction = interpolate_quantiles(requested_quantiles, self.config.quantile_levels, native)

        if horizon > step_horizon:
            paths = self._expand_prediction_paths(buffer, len(requested_quantiles))
            predictions = [prediction]
            decoded = step_horizon
            remaining = horizon - step_horizon
            while remaining > 0:
                previous_width = predictions[-1].shape[1]
                paths = self._update_buffer_with_predictions(
                    paths,
                    predictions[-1],
                    at=context_width + decoded - previous_width,
                )
                step_horizon = min(_round_up(remaining, self.patch_size), self.max_horizon)
                window = paths.time_slice(decoded, context_width + decoded + step_horizon)
                native = self._predict_step(window, step_horizon)[: target_count * len(requested_quantiles)]
                native = native.reshape(
                    target_count,
                    len(requested_quantiles),
                    step_horizon,
                    self.head.n_quantiles,
                ).transpose(0, 1, 3, 2)
                prediction = reduce_rollout_quantiles(
                    native,
                    self.config.quantile_levels,
                    requested_quantiles,
                )
                predictions.append(prediction)
                decoded += step_horizon
                remaining -= step_horizon
            prediction = mx.concatenate(predictions, axis=1)

        predictions = _sanitize_predictions(prediction[:, :horizon])
        if context_array.ndim == 3:
            predictions = predictions.reshape(
                context_array.shape[0],
                context_array.shape[1],
                horizon,
                len(requested_quantiles),
            )
        mx.eval(predictions)
        return Forecast(predictions, requested_quantiles)

    def _predict_step(self, window: TimeSeries, horizon: int) -> mx.array:
        if self._compiled_single_pass is None:
            predictions = self._single_pass_arrays(
                window.variates,
                window.mask,
                window.group_ids,
                window.variate_type,
            )
        else:
            predictions = self._compiled_single_pass(
                window.variates,
                window.mask,
                window.group_ids,
                window.variate_type,
            )
        context_patches = (window.seq_len - horizon) // self.patch_size
        horizon_patches = horizon // self.patch_size
        predictions = predictions[:, context_patches - 1 : context_patches - 1 + horizon_patches]
        return predictions.reshape(predictions.shape[0], horizon, self.head.n_quantiles)

    def _expand_prediction_paths(self, buffer: TimeSeries, n_paths: int) -> TimeSeries:
        rows, width = buffer.variates.shape
        expanded_values = mx.repeat(buffer.variates[:, None, :], n_paths, axis=1).reshape(rows * n_paths, width)
        expanded_mask = mx.repeat(buffer.mask[:, None, :], n_paths, axis=1).reshape(rows * n_paths, width)
        row_groups = buffer.group_ids[:, -1]
        path_groups = (row_groups[:, None] * n_paths + mx.arange(n_paths)[None, :]).reshape(-1)
        expanded_groups = mx.broadcast_to(path_groups[:, None], (rows * n_paths, width))
        row_types = buffer.variate_type[:, -1]
        path_types = mx.repeat(row_types, n_paths)
        expanded_types = mx.broadcast_to(path_types[:, None], (rows * n_paths, width))
        return TimeSeries(expanded_values, expanded_mask, expanded_groups, expanded_types)

    def _update_buffer_with_predictions(self, buffer: TimeSeries, prediction: mx.array, at: int) -> TimeSeries:
        targets, horizon, n_paths = prediction.shape
        target_path_rows = targets * n_paths
        flattened = prediction.transpose(0, 2, 1).reshape(target_path_rows, horizon)
        target_values = mx.concatenate(
            [buffer.variates[:target_path_rows, :at], flattened, buffer.variates[:target_path_rows, at + horizon :]],
            axis=1,
        )
        target_mask = mx.concatenate(
            [
                buffer.mask[:target_path_rows, :at],
                mx.full((target_path_rows, horizon), MaskType.VALID, dtype=mx.int8),
                buffer.mask[:target_path_rows, at + horizon :],
            ],
            axis=1,
        )
        return TimeSeries(
            variates=mx.concatenate([target_values, buffer.variates[target_path_rows:]], axis=0),
            mask=mx.concatenate([target_mask, buffer.mask[target_path_rows:]], axis=0),
            group_ids=buffer.group_ids,
            variate_type=buffer.variate_type,
        )

    def _single_pass_arrays(
        self,
        variates: mx.array,
        mask: mx.array,
        group_ids: mx.array,
        variate_type: mx.array,
    ) -> mx.array:
        buffer = TimeSeries(variates, mask, group_ids, variate_type)
        scaled, loc_scale = self.scaler.scale_input(buffer)
        return self.scaler.rescale_predictions(self(scaled), loc_scale, self.patch_size)

    def _prepare_single_pass_buffer(
        self,
        context: TimeSeries,
        prediction_width: int,
        context_length: int,
    ) -> TimeSeries:
        rows = context.variates.shape[0]
        left_pad = (-context_length) % self.patch_size
        row_groups = context.group_ids[:, :1]
        row_types = context.variate_type[:, :1]
        target_rows = row_types[:, 0] == VariateType.TARGET
        future_rows = row_types[:, 0] == VariateType.FUTURE
        known = min(context.seq_len - context_length, prediction_width)

        forecast_values = mx.zeros((rows, prediction_width), dtype=mx.float32)
        forecast_mask = mx.full((rows, prediction_width), MaskType.PAD, dtype=mx.int8)
        forecast_mask = mx.where(target_rows[:, None], MaskType.WITHHELD, forecast_mask).astype(mx.int8)
        if known > 0:
            known_columns = mx.arange(prediction_width)[None, :] < known
            copy_future = future_rows[:, None] & known_columns
            source_values = context.variates[:, context_length : context_length + known]
            source_mask = context.mask[:, context_length : context_length + known]
            padded_values = mx.concatenate(
                [source_values, mx.zeros((rows, prediction_width - known), dtype=mx.float32)],
                axis=1,
            )
            padded_mask = mx.concatenate(
                [source_mask, mx.full((rows, prediction_width - known), MaskType.PAD, dtype=mx.int8)],
                axis=1,
            )
            forecast_values = mx.where(copy_future, padded_values, forecast_values)
            forecast_mask = mx.where(copy_future, padded_mask, forecast_mask)
        return TimeSeries(
            variates=mx.concatenate(
                [
                    mx.zeros((rows, left_pad), dtype=mx.float32),
                    context.variates[:, :context_length],
                    forecast_values,
                ],
                axis=1,
            ),
            mask=mx.concatenate(
                [
                    mx.full((rows, left_pad), MaskType.PAD, dtype=mx.int8),
                    context.mask[:, :context_length],
                    forecast_mask,
                ],
                axis=1,
            ),
            group_ids=mx.concatenate(
                [
                    mx.full((rows, left_pad), -1, dtype=mx.int32),
                    context.group_ids[:, :context_length],
                    mx.broadcast_to(row_groups, (rows, prediction_width)),
                ],
                axis=1,
            ),
            variate_type=mx.concatenate(
                [
                    mx.full((rows, left_pad), -1, dtype=mx.int32),
                    context.variate_type[:, :context_length],
                    mx.broadcast_to(row_types, (rows, prediction_width)),
                ],
                axis=1,
            ),
        )
