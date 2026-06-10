# Copyright 2026 The Forecasting Company
# The rollout logic of extending quantile predictions in different paths,
# to then reduce them through quantile projection
# follows Chronos-2's inference pipeline
# (https://github.com/amazon-science/chronos-forecasting,
# src/chronos/chronos2/pipeline.py).
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auto-regressive quantile rollouts.
The model can predict multiple time steps in parallel up to a maximum horizon.
Beyond this horizon, the prediction mechanism falls back to an auto-regressive rollout strategy.
"""

import logging
from typing import Protocol

import torch
from einops import rearrange, repeat
from jaxtyping import Float, Int
from torch import Tensor

from t0.data import MaskType, TimeSeries, VariateType
from t0.model.layers import QuantileHead
from t0.quantile import QuantileRolloutReducer, interpolate_quantiles
from t0.scaler import CausalScaler

logger = logging.getLogger(__name__)


def _round_up(value: int, multiple: int) -> int:
    """Smallest multiple of ``multiple`` that is ``>= value``."""
    return -(-value // multiple) * multiple


class RolloutModel(Protocol):
    """Structural type of the model the rollout drives."""

    patch_size: int
    max_horizon: int
    scaler: CausalScaler
    head: QuantileHead

    def __call__(self, model_input: TimeSeries) -> Float[Tensor, "variates patches patch_size quantiles"]:
        """Per-patch quantile predictions for every patch position."""


class RolloutManager:
    """Predicts up to ``max_horizon`` steps in one pass; beyond, falls back to an auto-regressive rollout."""

    def __init__(self, model: RolloutModel):
        self.model = model

    def predict(
        self,
        batch: TimeSeries,
        prediction_length: int,
        query_quantile_levels: Float[Tensor, " query_quantiles"],
        context_length: int,
    ) -> Float[Tensor, "targets prediction_length query_quantiles"]:
        """Forecast ``prediction_length`` steps for every target row of ``batch``.

        Columns of ``batch`` past ``context_length`` are known future covariates.
        """
        model = self.model
        patch_size = model.patch_size
        if model.max_horizon < patch_size or model.max_horizon % patch_size != 0:
            raise ValueError(
                f"max_horizon must be a positive multiple of patch_size ({patch_size}), got {model.max_horizon}"
            )

        target_rows = batch.variate_type[:, 0] == VariateType.TARGET
        buffer = self.prepare_rollout_buffer(batch, prediction_length, context_length)
        context_width = _round_up(context_length, patch_size)

        horizon = min(_round_up(prediction_length, patch_size), model.max_horizon)
        block = self.predict_step(buffer.time_slice(0, context_width + horizon), horizon)[target_rows]
        prediction = interpolate_quantiles(query_quantile_levels, model.head.quantile_levels, block)
        if prediction_length <= horizon:
            return prediction[:, :prediction_length]

        logger.debug(
            "prediction_length %d exceeds max_horizon %d — continuing autoregressively",
            prediction_length,
            model.max_horizon,
        )
        n_paths = len(query_quantile_levels)
        paths = self.expand_prediction_paths(buffer, n_paths)
        path_target_rows = paths.variate_type[:, 0] == VariateType.TARGET
        reducer = QuantileRolloutReducer(
            predicted_quantile_levels=model.head.quantile_levels,
            query_quantile_levels=query_quantile_levels,
        )

        predictions = [prediction]
        decoded = horizon
        remaining = prediction_length - horizon
        while remaining > 0:
            prev_width = predictions[-1].shape[1]
            paths = self.update_buffer_with_predictions(paths, predictions[-1], at=context_width + decoded - prev_width)
            horizon = min(_round_up(remaining, patch_size), model.max_horizon)
            window = paths.time_slice(decoded, context_width + decoded + horizon)
            block = self.predict_step(window, horizon)[path_target_rows]
            prediction = reducer.reduce(rearrange(block, "(t q) h pq -> t q pq h", q=n_paths))
            predictions.append(prediction)
            decoded += horizon
            remaining -= horizon
        return torch.cat(predictions, dim=1)[:, :prediction_length]

    def prepare_rollout_buffer(self, batch: TimeSeries, prediction_length: int, context_length: int) -> TimeSeries:
        """Build the rollout buffer: padded context + a forecast region (targets WITHHELD, known futures VALID)."""
        patch_size = self.model.patch_size
        device = batch.device
        n_rows = batch.variates.shape[0]
        pad_left = (-context_length) % patch_size
        forecast_width = _round_up(prediction_length, patch_size)
        known = min(batch.seq_len - context_length, forecast_width)  # future cols already in `batch`

        forecast_values = torch.zeros((n_rows, forecast_width), dtype=batch.variates.dtype, device=device)
        forecast_mask = torch.full((n_rows, forecast_width), MaskType.PAD, dtype=torch.int8, device=device)
        forecast_mask[batch.variate_type[:, 0] == VariateType.TARGET] = MaskType.WITHHELD
        if known > 0:
            future_rows = batch.variate_type[:, 0] == VariateType.FUTURE
            forecast_values[future_rows, :known] = batch.variates[future_rows, context_length : context_length + known]
            forecast_mask[future_rows, :known] = batch.mask[future_rows, context_length : context_length + known]
        row_group = batch.group_ids[:, :1].expand(n_rows, forecast_width)
        row_type = batch.variate_type[:, :1].expand(n_rows, forecast_width)

        pad_values = torch.zeros((n_rows, pad_left), dtype=batch.variates.dtype, device=device)
        pad_mask = torch.full((n_rows, pad_left), MaskType.PAD, dtype=torch.int8, device=device)
        pad_sentinel = torch.full((n_rows, pad_left), -1, dtype=torch.long, device=device)
        context = slice(0, context_length)
        return TimeSeries(
            variates=torch.cat([pad_values, batch.variates[:, context], forecast_values], dim=1),
            mask=torch.cat([pad_mask, batch.mask[:, context], forecast_mask], dim=1),
            group_ids=torch.cat([pad_sentinel, batch.group_ids[:, context], row_group], dim=1),
            variate_type=torch.cat([pad_sentinel, batch.variate_type[:, context], row_type], dim=1),
        )

    def expand_prediction_paths(self, buffer: TimeSeries, n_paths: int) -> TimeSeries:
        """Replicate each row into ``n_paths`` trajectories with distinct group ids, one per query quantile."""
        n_rows, width = buffer.variates.shape
        row_group_ids: Int[Tensor, " variates"] = buffer.group_ids[:, -1]
        row_variate_type: Int[Tensor, " variates"] = buffer.variate_type[:, -1]
        path_offsets = torch.arange(n_paths, device=buffer.device)
        group_ids = rearrange(row_group_ids.unsqueeze(1) * n_paths + path_offsets, "v q -> (v q)")
        return TimeSeries(
            variates=repeat(buffer.variates, "v s -> (v q) s", q=n_paths),
            mask=repeat(buffer.mask, "v s -> (v q) s", q=n_paths),
            group_ids=repeat(group_ids, "vq -> vq s", s=width).contiguous(),
            variate_type=repeat(row_variate_type, "v -> (v q) s", q=n_paths, s=width).contiguous(),
        )

    def update_buffer_with_predictions(
        self,
        buffer: TimeSeries,
        prediction: Float[Tensor, "targets horizon query_quantiles"],
        at: int,
    ) -> TimeSeries:
        """Write target predictions into the forecast region (``VALID``); future rows are left untouched."""
        target_rows = buffer.variate_type[:, 0] == VariateType.TARGET
        horizon = prediction.shape[1]
        variates = buffer.variates.clone()
        mask = buffer.mask.clone()
        variates[target_rows, at : at + horizon] = rearrange(prediction, "t h q -> (t q) h").to(variates.dtype)
        mask[target_rows, at : at + horizon] = MaskType.VALID
        return TimeSeries(variates=variates, mask=mask, group_ids=buffer.group_ids, variate_type=buffer.variate_type)

    def predict_step(self, window: TimeSeries, horizon: int) -> Float[Tensor, "variates horizon model_quantiles"]:
        """Scale, run the model, and rescale one window; return the ``horizon`` patches after the context end."""
        model = self.model
        scaled, loc_scale = model.scaler.scale_input(window)
        predictions = model.scaler.rescale_predictions(model(scaled), loc_scale, model.patch_size)
        context_patches = (window.seq_len - horizon) // model.patch_size
        return predictions[:, context_patches - 1 : context_patches - 1 + horizon // model.patch_size].flatten(1, 2)
