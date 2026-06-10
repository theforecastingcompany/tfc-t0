# Copyright 2026 The Forecasting Company
# The Welford-style cumulative causal statistics borrow from Datadog's Toto
# scaler (https://github.com/DataDog/toto).
# Copyright 2025 Datadog, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Causal scaler for time series normalization."""

import logging
from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from t0.data import TimeSeries, VariateType

logger = logging.getLogger(__name__)
# Same epsilon as upstream Toto: https://github.com/DataDog/toto/blob/b4d4c9f3e121701fb02f65d525a435bf551d2582/toto/model/scaler.py#L307
EPS = 1e-1


@dataclass
class LocScale:
    """Container for scaling statistics."""

    loc: Float[Tensor, "variates patches"]
    scale: Float[Tensor, "variates patches"]

    def __post_init__(self) -> None:
        if self.loc.shape != self.scale.shape:
            raise ValueError(f"Loc and scale must have the same shape, got {self.loc.shape} and {self.scale.shape}")
        if self.loc.isnan().any() or self.scale.isnan().any():
            raise ValueError("Loc and scale must not contain NaN values")
        if (self.scale < 0).any():
            raise ValueError("Scale must be positive")


def _segmented_cumsum(
    x: Float[Tensor, "variates time"], boundary: Bool[Tensor, "variates time"]
) -> Float[Tensor, "variates time"]:
    """Cumulative sum along the last dim that resets at every ``True`` cell of ``boundary``."""
    full_cs = torch.cumsum(x, dim=-1)
    segment_ids = boundary.long().cumsum(dim=-1) - 1

    offsets_raw = torch.zeros_like(x)
    offsets_raw[..., 1:] = full_cs[..., :-1]
    offsets_at_boundaries = offsets_raw * boundary.to(x.dtype)

    leading_shape = x.shape[:-1]
    max_segs = int(segment_ids.max().item()) + 1
    per_seg_offset = torch.zeros(*leading_shape, max_segs, device=x.device, dtype=x.dtype)
    per_seg_offset.scatter_add_(-1, segment_ids, offsets_at_boundaries)

    offset = per_seg_offset.gather(-1, segment_ids)
    return full_cs - offset


def _group_ids_to_boundary(group_ids: Int[Tensor, "variates time"]) -> Bool[Tensor, "variates time"]:
    """Boundary mask: ``True`` at every cell where ``group_ids`` changes value."""
    _v, t = group_ids.shape
    boundary = torch.ones_like(group_ids, dtype=torch.bool)
    if t > 1:
        boundary[:, 1:] = group_ids[:, 1:] != group_ids[:, :-1]
    return boundary


def _compute_causal_stats(
    x: Float[Tensor, "variates time"],
    mask: Bool[Tensor, "variates time"] | None,
    group_ids: Int[Tensor, "variates time"],
) -> tuple[Float[Tensor, "variates time"], Float[Tensor, "variates time"]]:
    """Welford causal mean/std along the last dim, resetting at ``group_ids`` boundaries.

    For inference inputs (one independent series per row → one segment per
    row) this reduces to plain per-row cumulative Welford. The segment-aware
    machinery is preserved so the same code path serves both flavours.
    """
    boundary = _group_ids_to_boundary(group_ids)

    invalid = torch.zeros_like(x, dtype=torch.bool) if mask is None else mask.clone()
    invalid |= torch.isnan(x)
    valid = ~invalid

    cumcount = _segmented_cumsum(valid.to(x.dtype), boundary)
    cumcount_safe = cumcount.clamp(min=1.0)

    masked_x = x.masked_fill(invalid, 0.0)
    cumsum_x = _segmented_cumsum(masked_x, boundary)

    means = cumsum_x / cumcount_safe

    shifted_means = torch.zeros_like(means)
    shifted_means[:, 1:] = means[:, :-1]
    if boundary.shape[-1] > 1:
        shifted_means[:, 1:][boundary[:, 1:]] = 0.0

    delta = masked_x - shifted_means
    increment = delta * (masked_x - means) * valid

    m_2 = _segmented_cumsum(increment, boundary).clamp(min=0.0)
    variance = m_2 / (cumcount_safe - 1.0).clamp(min=1.0)
    stds = torch.sqrt(variance + EPS)
    return means, stds


def _compute_global_stats(
    x: Float[Tensor, "variates time"],
    mask: Bool[Tensor, "variates time"],
    group_ids: Int[Tensor, "variates time"],
) -> tuple[Float[Tensor, "variates time"], Float[Tensor, "variates time"]]:
    """Per-segment global (non-causal) mean and std, broadcast back to ``(V, T)``."""
    boundary = _group_ids_to_boundary(group_ids)
    segment_ids = boundary.long().cumsum(dim=-1) - 1
    max_segments = int(segment_ids.max().item()) + 1

    v = x.shape[0]
    device = x.device
    dtype = x.dtype

    invalid = mask | torch.isnan(x)
    valid = ~invalid
    masked_x = x.masked_fill(invalid, 0.0)

    seg_sum = torch.zeros(v, max_segments, device=device, dtype=dtype)
    seg_count = torch.zeros(v, max_segments, device=device, dtype=dtype)
    seg_sum.scatter_add_(1, segment_ids, masked_x)
    seg_count.scatter_add_(1, segment_ids, valid.to(dtype))

    seg_mean = seg_sum / seg_count.clamp(min=1.0)
    pos_mean = seg_mean.gather(1, segment_ids)

    squared_diff = ((x - pos_mean) ** 2).masked_fill(invalid, 0.0)
    seg_sq_sum = torch.zeros(v, max_segments, device=device, dtype=dtype)
    seg_sq_sum.scatter_add_(1, segment_ids, squared_diff)
    seg_var = seg_sq_sum / seg_count.clamp(min=2.0)
    seg_std = seg_var.sqrt().clamp(min=EPS)
    pos_std = seg_std.gather(1, segment_ids)
    return pos_mean, pos_std


class CausalScaler(torch.nn.Module):
    """Per-row causal scaler used by ``T0Forecaster``.

    Targets and historicals get causal stats (Welford); futures get
    per-row global stats. Optionally applies arcsinh after the standard
    ``(x - loc) / scale`` step (T0-novel, helps with extreme outliers).

    Stateless: zero parameters, zero buffers, contributes nothing to
    ``state_dict``.
    """

    def __init__(self, patch_size: int = 1, use_arcsinh: bool = False):
        super().__init__()
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        self.patch_size = patch_size
        self.use_arcsinh = use_arcsinh

    def scale_input(
        self,
        grouped_input: TimeSeries,
    ) -> tuple[TimeSeries, LocScale]:
        variates = grouped_input.variates
        group_ids = grouped_input.group_ids
        variate_type = grouped_input.variate_type
        invalid = ~grouped_input.valid_mask
        v, t = variates.shape

        non_padding = group_ids >= 0
        is_causal = non_padding & ((variate_type == VariateType.TARGET) | (variate_type == VariateType.HISTORICAL))
        is_future = non_padding & (variate_type == VariateType.FUTURE)

        loc = torch.zeros(v, t, device=variates.device, dtype=variates.dtype)
        scale = torch.ones(v, t, device=variates.device, dtype=variates.dtype)

        if is_causal.any():
            causal_loc, causal_scale = _compute_causal_stats(variates, mask=invalid, group_ids=group_ids)
            loc = torch.where(is_causal, causal_loc, loc)
            scale = torch.where(is_causal, causal_scale, scale)
        if is_future.any():
            future_loc, future_scale = _compute_global_stats(variates, mask=invalid, group_ids=group_ids)
            loc = torch.where(is_future, future_loc, loc)
            scale = torch.where(is_future, future_scale, scale)

        # Subsample to one stat per patch (the right edge of each patch).
        fcd_loc = loc[:, self.patch_size - 1 :: self.patch_size]
        fcd_scale = scale[:, self.patch_size - 1 :: self.patch_size]

        is_padding_fcd = group_ids[:, self.patch_size - 1 :: self.patch_size] < 0
        fcd_loc = fcd_loc.masked_fill(is_padding_fcd, 0.0)
        fcd_scale = fcd_scale.masked_fill(is_padding_fcd, 1.0)
        loc_scale = LocScale(loc=fcd_loc, scale=fcd_scale)

        # Re-expand stats to per-cell to apply the standardisation.
        loc_expanded = loc_scale.loc.repeat_interleave(self.patch_size, dim=-1)
        scale_expanded = loc_scale.scale.repeat_interleave(self.patch_size, dim=-1)
        scaled_variates = (variates - loc_expanded) / scale_expanded
        if self.use_arcsinh:
            scaled_variates = torch.arcsinh(scaled_variates)

        scaled_input = TimeSeries(
            variates=scaled_variates,
            group_ids=grouped_input.group_ids,
            variate_type=grouped_input.variate_type,
            mask=grouped_input.mask,
        )
        return scaled_input, loc_scale

    def rescale_predictions(
        self,
        predictions: Float[Tensor, "variates n_fcds *event"],
        loc_scale: LocScale,
        model_patch_size: int,
    ) -> Float[Tensor, "variates n_fcds *event"]:
        """Inverse-transform predictions back to data space."""
        n_fcds = predictions.shape[1]
        fcd_loc_scale = self._subsample_to_fcds(loc_scale, n_fcds, model_patch_size)
        return self._apply_inverse_scaling(predictions, fcd_loc_scale)

    def _subsample_to_fcds(
        self,
        loc_scale: LocScale,
        n_fcds: int,
        model_patch_size: int,
    ) -> LocScale:
        n_scaler_patches = loc_scale.loc.shape[1]
        if n_scaler_patches == n_fcds:
            return loc_scale
        if n_scaler_patches % model_patch_size != 0:
            raise ValueError(
                f"loc_scale n_patches ({n_scaler_patches}) must be divisible by "
                f"model_patch_size ({model_patch_size}) for clean FCD extraction"
            )
        expected_fcds = n_scaler_patches // model_patch_size
        if expected_fcds != n_fcds:
            raise ValueError(
                f"Slicing would produce {expected_fcds} FCDs but expected {n_fcds}. "
                f"Check n_scaler_patches ({n_scaler_patches}) and model_patch_size ({model_patch_size})."
            )
        loc = loc_scale.loc[:, model_patch_size - 1 :: model_patch_size]
        scale = loc_scale.scale[:, model_patch_size - 1 :: model_patch_size]
        return LocScale(loc=loc, scale=scale)

    def _apply_inverse_scaling(
        self,
        x: Float[Tensor, "variates n_fcds *event"],
        loc_scale: LocScale,
    ) -> Float[Tensor, "variates n_fcds *event"]:
        n_fcds = x.shape[1]
        if loc_scale.loc.shape[1] != n_fcds:
            raise ValueError(f"loc_scale n_patches ({loc_scale.loc.shape[1]}) must match x n_fcds ({n_fcds})")
        loc = loc_scale.loc
        scale = loc_scale.scale
        for _ in range(x.ndim - 2):
            loc = loc.unsqueeze(-1)
            scale = scale.unsqueeze(-1)
        if self.use_arcsinh:
            x = torch.sinh(x)
        return x * scale + loc
