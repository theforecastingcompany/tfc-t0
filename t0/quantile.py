# Copyright 2026 The Forecasting Company
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# The weighted-quantile and probability-mass computations are adapted from
# Chronos-2 (https://github.com/amazon-science/chronos-forecasting,
# src/chronos/utils.py and src/chronos/chronos2/pipeline.py).

"""Pure-math quantile utilities for rollout reduction and interpolation.

The ``QuantileRolloutReducer`` math (per-path empirical CDF + reduction
back to the trained quantile grid) follows Chronos-2's pipeline; see
<https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py>.

Used by ``T0Forecaster.predict()``'s autoregressive rollout.
"""

from dataclasses import dataclass, field

import torch
from einops import rearrange, repeat
from jaxtyping import Float

__all__ = [
    "interpolate_quantiles",
    "weighted_quantile",
    "get_prob_mass_per_quantile_level",
    "QuantileRolloutReducer",
]


def interpolate_quantiles(
    query_quantile_levels: Float[torch.Tensor, " query_quantiles"],
    original_quantile_levels: Float[torch.Tensor, "*batch orig_quantiles"],
    original_values: Float[torch.Tensor, "*batch orig_quantiles"],
) -> Float[torch.Tensor, "*batch query_quantiles"]:
    """Interpolate quantile values at specified query levels using linear interpolation."""
    assert torch.is_floating_point(original_values), "`original_values` must be a floating point tensor"
    orig_dtype = original_values.dtype
    if isinstance(query_quantile_levels, list):
        query_quantile_levels = torch.tensor(query_quantile_levels, dtype=torch.float32)
    if isinstance(original_quantile_levels, list):
        original_quantile_levels = torch.tensor(original_quantile_levels, dtype=torch.float32)

    assert query_quantile_levels.ndim == 1, "`query_quantile_levels` must be 1-dimensional"
    if original_quantile_levels.ndim > 1:
        assert original_quantile_levels.shape == original_values.shape, (
            "If `original_quantile_levels` is not 1D, its shape must match `original_values`"
        )
    else:
        assert len(original_quantile_levels) == original_values.shape[-1], (
            "If `original_quantile_levels` is 1D, its length must match the last dim of `original_values`"
        )
    assert query_quantile_levels.min() >= 0.0 and query_quantile_levels.max() <= 1.0, (
        "`query_quantile_levels` must be between 0 and 1"
    )
    assert original_quantile_levels.min() >= 0.0 and original_quantile_levels.max() <= 1.0, (
        "`original_quantile_levels` must be between 0 and 1"
    )
    original_quantile_levels = torch.clamp(original_quantile_levels, min=0.0, max=1.0)

    device = original_values.device
    query_quantile_levels = query_quantile_levels.to(device)
    original_quantile_levels = original_quantile_levels.to(device)
    original_values = original_values.to(torch.float32)

    orig_values_shape = original_values.shape
    num_original_quantiles = original_quantile_levels.shape[-1]
    original_values = original_values.reshape(-1, num_original_quantiles)
    batch_size = original_values.shape[0]

    if original_quantile_levels.ndim == 1:
        original_quantile_levels = original_quantile_levels.expand(batch_size, -1)
    else:
        original_quantile_levels = original_quantile_levels.reshape(-1, num_original_quantiles)

    sorted_levels, sorted_indices = torch.sort(original_quantile_levels, dim=-1)
    sorted_values = torch.gather(original_values, dim=-1, index=sorted_indices)

    zeros_padding = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
    ones_padding = torch.ones((batch_size, 1), dtype=torch.float32, device=device)

    sorted_levels_with_padding = []
    sorted_values_with_padding = []
    if original_quantile_levels.min() > 0.0:
        sorted_levels_with_padding.append(zeros_padding)
        sorted_values_with_padding.append(sorted_values[:, :1])
    sorted_levels_with_padding.append(sorted_levels)
    sorted_values_with_padding.append(sorted_values)
    if original_quantile_levels.max() < 1.0:
        sorted_levels_with_padding.append(ones_padding)
        sorted_values_with_padding.append(sorted_values[:, -1:])

    sorted_levels = torch.cat(sorted_levels_with_padding, dim=-1).contiguous()
    sorted_values = torch.cat(sorted_values_with_padding, dim=-1)

    query_levels_expanded = repeat(query_quantile_levels, "q -> b q", b=batch_size).contiguous()

    upper_indices = torch.searchsorted(sorted_levels, query_levels_expanded, right=True)
    upper_indices = torch.clamp(upper_indices, max=sorted_levels.shape[-1] - 1)
    lower_indices = upper_indices - 1

    lower_levels = torch.gather(sorted_levels, dim=1, index=lower_indices)
    upper_levels = torch.gather(sorted_levels, dim=1, index=upper_indices)
    lower_values = torch.gather(sorted_values, dim=1, index=lower_indices)
    upper_values = torch.gather(sorted_values, dim=1, index=upper_indices)

    level_diff = upper_levels - lower_levels
    weight = torch.nan_to_num((query_levels_expanded - lower_levels) / level_diff, nan=0.0)
    interpolated_values = lower_values + weight * (upper_values - lower_values)

    final_shape = (*orig_values_shape[:-1], len(query_quantile_levels))
    return interpolated_values.reshape(final_shape).to(orig_dtype)


def weighted_quantile(
    query_quantile_levels: Float[torch.Tensor, " query_quantiles"],
    sample_weights: Float[torch.Tensor, " num_samples"],
    samples: Float[torch.Tensor, "*batch num_samples"],
) -> Float[torch.Tensor, "*batch query_quantiles"]:
    """Compute quantiles from weighted samples using an empirical CDF.

    Adapted from Chronos-2:
    https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/utils.py#L135-L212
    """
    assert torch.is_floating_point(samples), "`samples` must be a floating point tensor"
    orig_dtype = samples.dtype
    if isinstance(query_quantile_levels, list):
        query_quantile_levels = torch.tensor(query_quantile_levels, dtype=torch.float32)
    if isinstance(sample_weights, list):
        sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    assert query_quantile_levels.ndim == 1 and sample_weights.ndim == 1, (
        "`query_quantile_levels` and `sample_weights` must be 1-dimensional"
    )
    assert len(sample_weights) == samples.shape[-1], (
        "the last dim of `samples` must be equal to the length of `sample_weights`"
    )
    assert query_quantile_levels.min() >= 0.0 and query_quantile_levels.max() <= 1.0, (
        "`query_quantile_levels` must be between 0 and 1"
    )
    assert sample_weights.min() > 0.0, "`sample_weights` must be > 0"

    device = samples.device
    query_quantile_levels = query_quantile_levels.to(device)
    sample_weights = sample_weights.to(device)
    samples = samples.to(torch.float32)

    orig_samples_shape = samples.shape
    num_samples = len(sample_weights)
    samples = samples.reshape(-1, num_samples)
    batch_size = samples.shape[0]

    sample_weights = sample_weights / sample_weights.sum(dim=-1, keepdim=True)
    sample_weights = sample_weights.expand(batch_size, -1).contiguous()

    sorted_samples, sort_indices = torch.sort(samples, dim=-1)
    sorted_weights = torch.gather(sample_weights, dim=-1, index=sort_indices)

    cumul_weights = torch.cumsum(sorted_weights, dim=-1)
    cumul_weights = torch.clamp(cumul_weights, min=0.0, max=1.0)

    interpolated_quantiles = interpolate_quantiles(
        query_quantile_levels=query_quantile_levels,
        original_quantile_levels=cumul_weights,
        original_values=sorted_samples,
    )

    final_shape = (*orig_samples_shape[:-1], len(query_quantile_levels))
    return interpolated_quantiles.reshape(final_shape).to(dtype=orig_dtype)


def get_prob_mass_per_quantile_level(
    quantile_levels: Float[torch.Tensor, " quantiles"],
) -> Float[torch.Tensor, " quantiles"]:
    """Compute normalized probability masses per quantile (trapezoidal rule).

    Adapted from Chronos-2:
    https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py#L48-L74
    """
    assert quantile_levels.ndim == 1
    assert quantile_levels.min() > 0.0 and quantile_levels.max() < 1.0

    device = quantile_levels.device
    boundaries = torch.cat([torch.tensor([0.0], device=device), quantile_levels, torch.tensor([1.0], device=device)])
    prob_mass = (boundaries[2:] - boundaries[:-2]) / 2
    return prob_mass / prob_mass.sum()


@dataclass
class QuantileRolloutReducer:
    """Reduce multi-step quantile predictions during autoregressive rollout.

    Approach is based on Chronos-2 pipeline's predict method:
    https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py#L450-L648
    """

    predicted_quantile_levels: Float[torch.Tensor, " predicted_quantiles"]
    query_quantile_levels: Float[torch.Tensor, " query_quantiles"]
    sample_weights: Float[torch.Tensor, " samples"] = field(init=False)

    def __post_init__(self):
        self.sample_weights = rearrange(
            torch.outer(
                get_prob_mass_per_quantile_level(self.predicted_quantile_levels),
                get_prob_mass_per_quantile_level(self.query_quantile_levels),
            ),
            "pq qq -> (pq qq)",
        )

    def reduce(
        self,
        predictions: Float[torch.Tensor, "variates query_quantiles predicted_quantiles horizon"],
    ) -> Float[torch.Tensor, "variates horizon query_quantiles"]:
        return weighted_quantile(
            query_quantile_levels=self.query_quantile_levels,
            sample_weights=self.sample_weights,
            samples=rearrange(predictions, "v qq pq h -> v h (pq qq)"),
        )
