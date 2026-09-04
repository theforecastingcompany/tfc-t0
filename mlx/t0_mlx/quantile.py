# Copyright 2026 The Forecasting Company
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# The weighted-quantile and probability-mass computations are adapted from
# Chronos-2 (https://github.com/amazon-science/chronos-forecasting,
# src/chronos/utils.py and src/chronos/chronos2/pipeline.py).
# SPDX-License-Identifier: Apache-2.0

"""Quantile interpolation and autoregressive rollout reduction."""

from bisect import bisect_right
from collections.abc import Sequence

import mlx.core as mx


def validate_quantiles(quantiles: Sequence[float]) -> tuple[float, ...]:
    """Validate and normalize a public quantile request."""
    levels = tuple(float(level) for level in quantiles)
    if not levels:
        raise ValueError("quantiles must be non-empty")
    if any(not 0.0 < level < 1.0 for level in levels):
        raise ValueError("each quantile must be in (0, 1)")
    if levels != tuple(sorted(set(levels))):
        raise ValueError("quantiles must be sorted ascending without duplicates")
    return levels


def interpolate_quantiles(query_levels: Sequence[float], source_levels: Sequence[float], values: mx.array) -> mx.array:
    """Linearly interpolate values and hold the endpoint quantiles constant."""
    source = tuple(float(level) for level in source_levels)
    columns: list[mx.array] = []
    for query in query_levels:
        upper = bisect_right(source, query)
        if upper == 0:
            columns.append(values[..., 0])
        elif upper == len(source):
            columns.append(values[..., -1])
        else:
            lower = upper - 1
            weight = (query - source[lower]) / (source[upper] - source[lower])
            columns.append(values[..., lower] + weight * (values[..., upper] - values[..., lower]))
    return mx.stack(columns, axis=-1)


def get_probability_mass(quantile_levels: Sequence[float]) -> mx.array:
    """Return normalized trapezoidal probability mass for quantile levels."""
    levels = mx.array(list(quantile_levels), dtype=mx.float32)
    boundaries = mx.concatenate([mx.array([0.0]), levels, mx.array([1.0])])
    mass = (boundaries[2:] - boundaries[:-2]) / 2.0
    return mass / mx.sum(mass)


def _interpolate_dynamic(query_levels: Sequence[float], levels: mx.array, values: mx.array) -> mx.array:
    """Interpolate when each flattened row has its own sorted level grid."""
    query = mx.array(list(query_levels), dtype=mx.float32)
    levels = mx.concatenate([mx.zeros_like(levels[:, :1]), levels, mx.ones_like(levels[:, :1])], axis=1)
    values = mx.concatenate([values[:, :1], values, values[:, -1:]], axis=1)

    upper = mx.sum(levels[:, :, None] <= query[None, None, :], axis=1).astype(mx.int32)
    upper = mx.minimum(upper, levels.shape[1] - 1)
    lower = upper - 1
    lower_levels = mx.take_along_axis(levels, lower, axis=1)
    upper_levels = mx.take_along_axis(levels, upper, axis=1)
    lower_values = mx.take_along_axis(values, lower, axis=1)
    upper_values = mx.take_along_axis(values, upper, axis=1)
    weight = mx.nan_to_num((query[None, :] - lower_levels) / (upper_levels - lower_levels), nan=0.0)
    return lower_values + weight * (upper_values - lower_values)


def weighted_quantile(query_levels: Sequence[float], sample_weights: mx.array, samples: mx.array) -> mx.array:
    """Reduce weighted samples through their empirical cumulative distribution."""
    original_shape = samples.shape
    n_samples = original_shape[-1]
    flattened = samples.reshape(-1, n_samples).astype(mx.float32)
    order = mx.argsort(flattened, axis=-1)
    sorted_samples = mx.take_along_axis(flattened, order, axis=-1)

    normalized_weights = sample_weights / mx.sum(sample_weights)
    weights = mx.broadcast_to(normalized_weights[None, :], flattened.shape)
    sorted_weights = mx.take_along_axis(weights, order, axis=-1)
    cumulative = mx.clip(mx.cumsum(sorted_weights, axis=-1), 0.0, 1.0)
    result = _interpolate_dynamic(query_levels, cumulative, sorted_samples)
    return result.reshape(*original_shape[:-1], len(query_levels)).astype(samples.dtype)


def reduce_rollout_quantiles(
    predictions: mx.array,
    predicted_levels: Sequence[float],
    query_levels: Sequence[float],
) -> mx.array:
    """Reduce `[targets, query_paths, native_quantiles, horizon]` to requested quantiles."""
    native_mass = get_probability_mass(predicted_levels)
    query_mass = get_probability_mass(query_levels)
    sample_weights = (native_mass[:, None] * query_mass[None, :]).reshape(-1)
    samples = predictions.transpose(0, 3, 2, 1).reshape(
        predictions.shape[0],
        predictions.shape[3],
        -1,
    )
    return weighted_quantile(query_levels, sample_weights, samples)
