"""MLX array representation of model inputs."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import mlx.core as mx


class VariateType(IntEnum):
    """Role of a variate in the model input."""

    TARGET = 0
    HISTORICAL = 1
    FUTURE = 2


class MaskType(IntEnum):
    """Reason that a time step is not an observed value."""

    VALID = 0
    PAD = 1
    MISSING = 2
    CENSORED = 3
    WITHHELD = 4


@dataclass
class TimeSeries:
    """Flat variate rows and their per-time-step metadata."""

    variates: mx.array
    mask: mx.array
    group_ids: mx.array
    variate_type: mx.array

    @property
    def seq_len(self) -> int:
        return self.variates.shape[1]

    @property
    def valid_mask(self) -> mx.array:
        return self.mask == MaskType.VALID

    def time_slice(self, start: int, stop: int) -> "TimeSeries":
        """Return a half-open slice along the time axis."""
        return TimeSeries(
            variates=self.variates[:, start:stop],
            mask=self.mask[:, start:stop],
            group_ids=self.group_ids[:, start:stop],
            variate_type=self.variate_type[:, start:stop],
        )

    @classmethod
    def from_context(
        cls,
        context: mx.array,
        mask: mx.array | None = None,
        group_ids: mx.array | None = None,
        future_covariates: mx.array | None = None,
        horizon: int = 0,
    ) -> "TimeSeries":
        """Build flat target and optional known-future covariate rows."""
        if context.ndim == 1:
            context = context[None, :]
            if mask is not None and mask.ndim == 1:
                mask = mask[None, :]
        if context.ndim not in (2, 3):
            raise ValueError(f"context must be [T], [B, T] or [B, V, T], got shape {context.shape}")
        if context.shape[-1] < 1:
            raise ValueError("context must contain at least one time step")

        batch_size = context.shape[0]
        n_variates = context.shape[1] if context.ndim == 3 else 1
        width = context.shape[-1]
        values = context.reshape(batch_size * n_variates, width)
        if mask is not None:
            if mask.shape != context.shape:
                raise ValueError(f"mask must have the same shape as context {context.shape}, got {mask.shape}")
            mask = mask.reshape(values.shape)

        is_nan = mx.isnan(values)
        if mask is None:
            mask = mx.where(is_nan, MaskType.MISSING, MaskType.VALID).astype(mx.int8)
        else:
            mask = mask.astype(mx.int8)
            if bool(mx.any((mask < MaskType.VALID) | (mask > MaskType.CENSORED)).item()):
                raise ValueError("mask values must be VALID, PAD, MISSING or CENSORED")
            if bool(mx.any(is_nan & (mask == MaskType.VALID)).item()):
                raise ValueError("mask marks NaN cells VALID; mark them MISSING or PAD")
        values = mx.where(is_nan, mx.array(0.0, dtype=context.dtype), values)

        rows = values.shape[0]
        if group_ids is None:
            row_groups = mx.repeat(mx.arange(batch_size, dtype=mx.int32), n_variates)
        else:
            if future_covariates is not None:
                raise ValueError("group_ids cannot be combined with future_covariates")
            if group_ids.ndim != 1 or group_ids.shape[0] != rows:
                raise ValueError(f"group_ids must hold one id per target row ({rows}), got {group_ids.shape}")
            group_ids = group_ids.astype(mx.int32)
            if bool(mx.any(group_ids < 0).item()):
                raise ValueError("group_ids must be non-negative")
            row_groups = group_ids
        row_groups = row_groups[:, None]
        group_ids = mx.broadcast_to(row_groups, (rows, width))
        variate_type = mx.full((rows, width), VariateType.TARGET, dtype=mx.int32)
        if future_covariates is None:
            return cls(values, mask, group_ids, variate_type)

        expected_width = width + horizon
        if (
            future_covariates.ndim != 3
            or future_covariates.shape[0] != batch_size
            or future_covariates.shape[2] != expected_width
        ):
            raise ValueError(
                f"future_covariates must be [B={batch_size}, F, T+horizon={expected_width}], "
                f"got shape {future_covariates.shape}"
            )
        if future_covariates.shape[1] == 0:
            return cls(values, mask, group_ids, variate_type)

        future_variates = future_covariates.shape[1]
        future_rows = batch_size * future_variates
        future_values = future_covariates.reshape(future_rows, expected_width)
        future_nan = mx.isnan(future_values)
        future_mask = mx.where(future_nan, MaskType.MISSING, MaskType.VALID).astype(mx.int8)
        future_values = mx.where(future_nan, mx.array(0.0, dtype=future_values.dtype), future_values)
        future_groups = mx.repeat(mx.arange(batch_size, dtype=mx.int32), future_variates)[:, None]
        future_groups = mx.broadcast_to(future_groups, (future_rows, expected_width))
        future_types = mx.full((future_rows, expected_width), VariateType.FUTURE, dtype=mx.int32)

        target_future_values = mx.zeros((rows, horizon), dtype=values.dtype)
        target_future_mask = mx.full((rows, horizon), MaskType.WITHHELD, dtype=mx.int8)
        target_future_groups = mx.broadcast_to(group_ids[:, :1], (rows, horizon))
        target_future_types = mx.broadcast_to(variate_type[:, :1], (rows, horizon))
        return cls(
            variates=mx.concatenate(
                [mx.concatenate([values, target_future_values], axis=1), future_values],
                axis=0,
            ),
            mask=mx.concatenate(
                [mx.concatenate([mask, target_future_mask], axis=1), future_mask],
                axis=0,
            ),
            group_ids=mx.concatenate(
                [mx.concatenate([group_ids, target_future_groups], axis=1), future_groups],
                axis=0,
            ),
            variate_type=mx.concatenate(
                [mx.concatenate([variate_type, target_future_types], axis=1), future_types],
                axis=0,
            ),
        )


def batch_series(series: Sequence[Any]) -> tuple[mx.array, mx.array, mx.array]:
    """Stack time series of potentially different lengths and variate counts into one model input.

    Each entry is one series, shaped ``(T,)`` or ``(V, T)``. Their variates are
    stacked along a single ``variates`` axis and right-aligned to the longest
    entry: cells that only widen a shorter entry are ``PAD``, NaN observations
    are ``MISSING``, and the returned group ids say which rows came from the
    same series, so its variates keep attending to one another.

    Raises:
        ValueError: ``series`` is empty, or an entry is not ``(T,)`` or ``(V, T)``.
    """
    entries = [mx.array(entry, dtype=mx.float32) for entry in series]
    if not entries:
        raise ValueError("series must hold at least one entry")
    entries = [entry[None, :] if entry.ndim == 1 else entry for entry in entries]
    if any(entry.ndim != 2 for entry in entries):
        raise ValueError("each series must be shaped [T] or [V, T]")
    if any(entry.shape[-1] < 1 for entry in entries):
        raise ValueError("each series must contain at least one time step")

    width = max(entry.shape[-1] for entry in entries)
    contexts = []
    masks = []
    groups = []
    for group, entry in enumerate(entries):
        pad = width - entry.shape[-1]
        contexts.append(mx.concatenate([mx.zeros((entry.shape[0], pad), dtype=mx.float32), entry], axis=1))
        observed_mask = mx.where(mx.isnan(entry), MaskType.MISSING, MaskType.VALID).astype(mx.int8)
        masks.append(
            mx.concatenate(
                [mx.full((entry.shape[0], pad), MaskType.PAD, dtype=mx.int8), observed_mask],
                axis=1,
            )
        )
        groups.append(mx.full((entry.shape[0],), group, dtype=mx.int32))
    return (
        mx.concatenate(contexts, axis=0),
        mx.concatenate(masks, axis=0),
        mx.concatenate(groups, axis=0),
    )
