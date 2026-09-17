# Copyright 2026 The Forecasting Company
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# The weighted-quantile and probability-mass computations are adapted from
# Chronos-2 (https://github.com/amazon-science/chronos-forecasting,
# src/chronos/utils.py and src/chronos/chronos2/pipeline.py).
# SPDX-License-Identifier: Apache-2.0

"""Quantile interpolation and autoregressive rollout reduction."""

import math
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


def get_rollout_quantile_levels(
    trained_quantile_levels: Sequence[float], requested_quantile_levels: Sequence[float]
) -> tuple[float, ...]:
    """Return the quantile levels a rollout must produce in order to serve a request.

    A requested level inside the trained range is returned as asked for. A level beyond
    it is left out and replaced by the trained levels its tail pins through: the
    boundary knot at that end and its inward neighbour. Interpolation would clamp such a
    level flat onto its boundary knot, so carrying it through the rollout would cost a
    path to reproduce a column the boundary knot already holds, and would reweight every
    other level through the reduction's probability mass.
    """
    knots = select_tail_knots(trained_quantile_levels, requested_quantile_levels)
    if knots is None:
        return tuple(requested_quantile_levels)
    interior = [level for level in requested_quantile_levels if knots[0] <= level <= knots[-1]]
    return tuple(sorted(set(interior).union(knots)))


def extrapolate_quantiles(
    query_quantile_levels: Sequence[float],
    original_quantile_levels: Sequence[float],
    original_values: mx.array,
    trained_quantile_levels: Sequence[float],
) -> mx.array:
    """Return quantiles at the query levels, extrapolating those beyond the trained range.

    ``original_values`` holds the quantiles at ``original_quantile_levels``, as returned
    by a rollout over ``get_rollout_quantile_levels``. Query levels inside the trained
    range are taken from it unchanged; levels outside it are evaluated on exponential
    tails pinned through the outermost trained levels.
    """
    knots = select_tail_knots(trained_quantile_levels, query_quantile_levels)
    if knots is None:
        return original_values
    left = [level for level in query_quantile_levels if level < knots[0]]
    right = [level for level in query_quantile_levels if level > knots[-1]]
    knot_columns = mx.take(original_values, mx.array([original_quantile_levels.index(knot) for knot in knots]), axis=-1)
    tail_values = extend_quantile_tails(left + right, knots, knot_columns)
    # The tails bracket the original levels, so the full grid is a plain concatenation.
    full = mx.concatenate([tail_values[..., : len(left)], original_values, tail_values[..., len(left) :]], axis=-1)
    full_levels = left + list(original_quantile_levels) + right
    return mx.take(full, mx.array([full_levels.index(level) for level in query_quantile_levels]), axis=-1)


def select_tail_knots(
    trained_quantile_levels: Sequence[float], requested_quantile_levels: Sequence[float]
) -> tuple[float, ...] | None:
    """Return the trained levels the exponential tails pin through, or None when no tail is needed.

    A tail pins through the boundary knot at its end and that knot's inward neighbour,
    so four trained levels at most are returned, or three when the range has an odd
    middle level shared by both ends.

    Trained levels are rounded to 6 decimals so float32 storage error
    (0.1 -> 0.10000000149...) cannot misclassify a requested level equal to a trained
    one as a tail.
    """
    ordered = sorted(round(float(level), 6) for level in trained_quantile_levels)
    if not any(level < ordered[0] or level > ordered[-1] for level in requested_quantile_levels):
        return None
    if len(ordered) < 2:
        raise ValueError("need at least two trained quantile levels to pin a tail")
    return tuple(sorted({ordered[0], ordered[1], ordered[-2], ordered[-1]}))


def extend_quantile_tails(
    tail_levels: Sequence[float], knot_levels: Sequence[float], knot_values: mx.array
) -> mx.array:
    """Evaluate IQF exponential tails (Park et al., arXiv 2111.06581) at levels outside the knot range.

    For knots k_0 < ... < k_N carrying values v_i = v(k_i), a level q below k_0 lies on
    the left tail, linear in log(q), and a level q above k_N on the right tail, linear
    in log(1 - q)::

        v(q) = v_0 + s_L * log(q / k_0)              s_L = (v_1 - v_0) / log(k_1 / k_0)
        v(q) = v_N + s_R * log((1 - k_N) / (1 - q))  s_R = (v_N - v_N-1) / log((1 - k_N-1) / (1 - k_N))

    Negative slopes (crossed knots) clamp to zero, degenerating to a flat tail, so both
    tails are continuous at the boundary knots and monotone by construction.
    """
    knots = [float(level) for level in knot_levels]
    left_slope = mx.maximum((knot_values[..., 1] - knot_values[..., 0]) / math.log(knots[1] / knots[0]), 0.0)
    right_slope = mx.maximum(
        (knot_values[..., -1] - knot_values[..., -2]) / math.log((1.0 - knots[-2]) / (1.0 - knots[-1])), 0.0
    )
    columns: list[mx.array] = []
    for level in tail_levels:
        if level < knots[0]:
            columns.append(knot_values[..., 0] + left_slope * math.log(level / knots[0]))
        else:
            columns.append(knot_values[..., -1] + right_slope * math.log((1.0 - knots[-1]) / (1.0 - level)))
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
