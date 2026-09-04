# Copyright 2026 The Forecasting Company
# The Welford-style cumulative causal statistics borrow from Datadog's Toto
# scaler (https://github.com/DataDog/toto).
# Copyright 2025 Datadog, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Causal scaling for the single-pass inference path."""

from dataclasses import dataclass

import mlx.core as mx

from t0_mlx.data import TimeSeries, VariateType

EPS = 1e-1


@dataclass
class LocScale:
    """Per-row, per-time-step location and scale."""

    loc: mx.array
    scale: mx.array


def _compute_causal_stats(x: mx.array, invalid: mx.array) -> tuple[mx.array, mx.array]:
    """Match the reference cumulative Welford calculation for one series per row."""
    valid = ~invalid
    counts = mx.cumsum(valid.astype(x.dtype), axis=-1)
    safe_counts = mx.maximum(counts, mx.array(1.0, dtype=x.dtype))
    masked_x = mx.where(invalid, mx.array(0.0, dtype=x.dtype), x)
    means = mx.cumsum(masked_x, axis=-1) / safe_counts

    shifted_means = mx.concatenate([mx.zeros_like(means[:, :1]), means[:, :-1]], axis=-1)
    delta = masked_x - shifted_means
    increments = delta * (masked_x - means) * valid.astype(x.dtype)
    m2 = mx.maximum(mx.cumsum(increments, axis=-1), mx.array(0.0, dtype=x.dtype))
    variance = m2 / mx.maximum(safe_counts - 1.0, mx.array(1.0, dtype=x.dtype))
    return means, mx.sqrt(variance + EPS)


def _compute_global_stats(x: mx.array, invalid: mx.array) -> tuple[mx.array, mx.array]:
    """Compute global population statistics independently for each row."""
    valid = ~invalid
    counts = mx.sum(valid.astype(x.dtype), axis=-1, keepdims=True)
    safe_counts = mx.maximum(counts, mx.array(1.0, dtype=x.dtype))
    masked = mx.where(invalid, mx.array(0.0, dtype=x.dtype), x)
    means = mx.sum(masked, axis=-1, keepdims=True) / safe_counts
    squared = mx.where(invalid, mx.array(0.0, dtype=x.dtype), mx.square(x - means))
    variance = mx.sum(squared, axis=-1, keepdims=True) / mx.maximum(
        counts,
        mx.array(2.0, dtype=x.dtype),
    )
    scales = mx.maximum(mx.sqrt(variance), mx.array(EPS, dtype=x.dtype))
    return mx.broadcast_to(means, x.shape), mx.broadcast_to(scales, x.shape)


class CausalScaler:
    """Per-row causal normalization used by the published checkpoint.

    The reference ``T0Forecaster`` uses scaler patch size 1, so targets and
    historicals retain one running statistic per timestep. Forecasts are
    inverse-scaled with the statistic at each model patch's right edge.
    """

    def __init__(self, use_arcsinh: bool = True):
        self.use_arcsinh = use_arcsinh

    def scale_input(self, model_input: TimeSeries) -> tuple[TimeSeries, LocScale]:
        invalid = ~model_input.valid_mask
        causal_loc, causal_scale = _compute_causal_stats(model_input.variates, invalid)
        future_loc, future_scale = _compute_global_stats(model_input.variates, invalid)
        non_padding = model_input.group_ids >= 0
        is_future = non_padding & (model_input.variate_type == VariateType.FUTURE)
        is_causal = non_padding & ~is_future
        loc = mx.where(is_causal, causal_loc, mx.zeros_like(causal_loc))
        scale = mx.where(is_causal, causal_scale, mx.ones_like(causal_scale))
        loc = mx.where(is_future, future_loc, loc)
        scale = mx.where(is_future, future_scale, scale)
        normalized = (model_input.variates - loc) / scale
        if self.use_arcsinh:
            normalized = mx.arcsinh(normalized)
        return (
            TimeSeries(
                variates=normalized,
                mask=model_input.mask,
                group_ids=model_input.group_ids,
                variate_type=model_input.variate_type,
            ),
            LocScale(loc=loc, scale=scale),
        )

    def rescale_predictions(self, predictions: mx.array, loc_scale: LocScale, model_patch_size: int) -> mx.array:
        loc = loc_scale.loc[:, model_patch_size - 1 :: model_patch_size]
        scale = loc_scale.scale[:, model_patch_size - 1 :: model_patch_size]
        for _ in range(predictions.ndim - 2):
            loc = mx.expand_dims(loc, axis=-1)
            scale = mx.expand_dims(scale, axis=-1)
        values = mx.sinh(predictions) if self.use_arcsinh else predictions
        return values * scale + loc
