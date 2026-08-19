"""Management of array representations of time series to be passed as input to the model."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


class VariateType(IntEnum):
    """Role of a variate."""

    TARGET = 0
    HISTORICAL = 1
    FUTURE = 2


class MaskType(IntEnum):
    """Reason of values for being masked.


    ``WITHHELD`` marks time steps the model must predict.
    """

    VALID = 0
    PAD = 1
    MISSING = 2
    CENSORED = 3
    WITHHELD = 4


def mask_nan_as_missing(values: Float[Tensor, "rows time"]) -> Int[Tensor, "rows time"]:
    """Read every NaN in ``values`` as an absent observation, and every other cell as valid.

    This is what a context without a mask means: NaN is missing data. Padding a
    shorter series is the case this cannot express — a padded cell holds no
    observation at all rather than a missing one, and only a mask can say so.
    """
    return torch.where(
        torch.isnan(values),
        torch.tensor(MaskType.MISSING, dtype=torch.int8, device=values.device),
        torch.tensor(MaskType.VALID, dtype=torch.int8, device=values.device),
    )


@dataclass
class TimeSeries:
    """Model input representing one or multiple time series whose variates are gathered in a tensor."""

    variates: Float[Tensor, "variates time"]
    mask: Int[Tensor, "variates time"]
    group_ids: Int[Tensor, "variates time"]
    variate_type: Int[Tensor, "variates time"]

    @cached_property
    def valid_mask(self) -> Bool[Tensor, "variates time"]:
        return self.mask == MaskType.VALID

    @property
    def device(self) -> torch.device:
        return self.variates.device

    @property
    def seq_len(self) -> int:
        return self.variates.shape[1]

    def to(self, device: torch.device) -> "TimeSeries":
        if device == self.device:
            return self
        return TimeSeries(
            variates=self.variates.to(device),
            mask=self.mask.to(device),
            group_ids=self.group_ids.to(device),
            variate_type=self.variate_type.to(device),
        )

    def time_slice(self, start: int, stop: int) -> "TimeSeries":
        """Return the ``[start, stop)`` window along the time axis."""
        return TimeSeries(
            variates=self.variates[:, start:stop],
            mask=self.mask[:, start:stop],
            group_ids=self.group_ids[:, start:stop],
            variate_type=self.variate_type[:, start:stop],
        )

    @staticmethod
    def _validate_mask(
        mask: Int[Tensor, "rows time"],
        shape: tuple[int, ...],
        is_nan: Bool[Tensor, "rows time"],
    ) -> None:
        """Raise unless ``mask`` describes every non-observed cell of a ``shape`` context."""
        if tuple(mask.shape) != shape:
            raise ValueError(f"mask must have the same shape as targets {shape}, got {tuple(mask.shape)}")
        # WITHHELD marks the region the model must predict; the rollout owns it.
        if not ((mask >= MaskType.VALID) & (mask <= MaskType.CENSORED)).all():
            raise ValueError("mask values must be MaskType.VALID, PAD, MISSING or CENSORED; WITHHELD is reserved")
        if (is_nan & (mask.reshape(is_nan.shape) == MaskType.VALID)).any():
            raise ValueError("mask marks NaN cells VALID; mark them MaskType.MISSING or MaskType.PAD")

    @staticmethod
    def _validate_group_ids(group_ids: Int[Tensor, " rows"], n_rows: int, has_future_covariates: bool) -> None:
        """Raise unless ``group_ids`` holds one usable id per target row."""
        if has_future_covariates:
            raise ValueError("group_ids cannot be combined with future_covariates")
        if group_ids.ndim != 1 or group_ids.shape[0] != n_rows:
            raise ValueError(f"group_ids must hold one id per target row ({n_rows}), got {tuple(group_ids.shape)}")
        # -1 is the padding sentinel the patcher and mask builder rely on.
        if (group_ids < 0).any():
            raise ValueError("group_ids must be non-negative")

    @classmethod
    def from_array(
        cls,
        targets: Float[Tensor, "batch time"] | Float[Tensor, "batch variates time"],
        future_covariates: Float[Tensor, "batch future_variates context_plus_horizon"] | None = None,
        mask: Int[Tensor, "batch time"] | Int[Tensor, "batch variates time"] | None = None,
        group_ids: Int[Tensor, " rows"] | None = None,
    ) -> "TimeSeries":
        """Build model input from a target context and optional future covariates.

        ``mask`` holds ``MaskType`` values shaped like ``targets``: ``MISSING``
        for an absent observation, ``PAD`` for a cell that only pads a shorter
        series out to the batch's width. Only all-``PAD`` patches leave attention.
        Without it every NaN in ``targets`` is read as an absent observation.

        ``group_ids`` holds one id per row of the flattened ``targets``; rows
        sharing an id are variates of one series and attend to one another.
        Without it every row of a ``(B, T)`` target is its own series.

        Raises:
            ValueError:
                - ``targets`` is not 2-/3-D.
                - ``future_covariates`` is not ``(B, F, >= T)``.
                - ``mask`` mismatches ``targets``' shape, sets ``WITHHELD``, or
                  marks a NaN cell ``VALID``.
                - ``group_ids`` is not one non-negative id per row, or comes
                  alongside ``future_covariates``.
        """
        if targets.ndim not in (2, 3):
            raise ValueError(f"targets must be (B, T) or (B, V, T), got shape {tuple(targets.shape)}")
        batch_size = targets.shape[0]
        n_variates = targets.shape[1] if targets.ndim == 3 else 1
        context_len = targets.shape[-1]
        device = targets.device
        sample_ids = torch.arange(batch_size, dtype=torch.long, device=device)

        n_target = batch_size * n_variates
        variates = targets.reshape(n_target, context_len)
        if mask is None:
            target_mask = mask_nan_as_missing(variates)
            # The helper writes MISSING on exactly the NaN cells, so reading them
            # back off the mask saves scanning `variates` for NaN a second time.
            is_nan = target_mask == MaskType.MISSING
        else:
            is_nan = torch.isnan(variates)
            cls._validate_mask(mask, tuple(targets.shape), is_nan)
            target_mask = mask.reshape(is_nan.shape).to(device=device, dtype=torch.int8)
        if is_nan.any():
            variates = torch.nan_to_num(variates, nan=0.0)
        if group_ids is None:
            row_groups = sample_ids.repeat_interleave(n_variates)
        else:
            cls._validate_group_ids(group_ids, n_target, future_covariates is not None)
            row_groups = group_ids.to(device=device, dtype=torch.long)
        row_group_ids = row_groups.unsqueeze(1).expand(n_target, context_len).contiguous()
        variate_type = torch.full((n_target, context_len), VariateType.TARGET, dtype=torch.long, device=device)
        if future_covariates is None or future_covariates.shape[1] == 0:
            return cls(variates=variates, mask=target_mask, group_ids=row_group_ids, variate_type=variate_type)

        total_len = future_covariates.shape[2]
        if future_covariates.ndim != 3 or future_covariates.shape[0] != batch_size or total_len < context_len:
            raise ValueError(
                f"future_covariates must be (B={batch_size}, F, T+H>=T={context_len}), "
                f"got shape {tuple(future_covariates.shape)}"
            )
        # Future rows span the full [0, T+H), VALID throughout (known context AND horizon).
        n_future = future_covariates.shape[1]
        rows = batch_size * n_future
        fut_values = future_covariates.to(device).reshape(rows, total_len)
        fut_mask = torch.full((rows, total_len), MaskType.VALID, dtype=torch.int8, device=device)
        fut_is_nan = torch.isnan(fut_values)
        if fut_is_nan.any():
            fut_mask = fut_mask.masked_fill(fut_is_nan, MaskType.MISSING)
            fut_values = torch.nan_to_num(fut_values, nan=0.0)
        fut_group = sample_ids.repeat_interleave(n_future).unsqueeze(1).expand(rows, total_len).contiguous()
        fut_type = torch.full((rows, total_len), VariateType.FUTURE, dtype=torch.long, device=device)

        # Extend target rows over the horizon with WITHHELD so all rows share one width.
        horizon = total_len - context_len
        h_values = torch.zeros((n_target, horizon), dtype=variates.dtype, device=device)
        h_mask = torch.full((n_target, horizon), MaskType.WITHHELD, dtype=torch.int8, device=device)
        h_group = row_group_ids[:, :1].expand(n_target, horizon)
        h_type = variate_type[:, :1].expand(n_target, horizon)
        return cls(
            variates=torch.cat([torch.cat([variates, h_values], dim=1), fut_values], dim=0),
            mask=torch.cat([torch.cat([target_mask, h_mask], dim=1), fut_mask], dim=0),
            group_ids=torch.cat([torch.cat([row_group_ids, h_group], dim=1), fut_group], dim=0),
            variate_type=torch.cat([torch.cat([variate_type, h_type], dim=1), fut_type], dim=0),
        )


def batch_series(
    series: Sequence[Float[Tensor, "*variates time"] | Float[np.ndarray, "*variates time"]],
) -> tuple[Float[Tensor, "variates time"], Int[Tensor, "variates time"], Int[Tensor, " variates"]]:
    """Stack time series of potentially different lengths and variate counts into one model input.

    Each entry is one series, shaped ``(T,)`` or ``(V, T)``. Their variates are
    stacked along a single ``variates`` axis and right-aligned to the longest
    entry: cells that only widen a shorter entry are ``PAD``, NaN observations
    are ``MISSING``, and the returned group ids say which rows came from the
    same series, so its variates keep attending to one another. The batch is
    built on the first entry's device.

    Raises:
        ValueError: ``series`` is empty, or an entry is not ``(T,)`` or ``(V, T)``.
    """
    rows = [torch.as_tensor(s, dtype=torch.float32) for s in series]
    if not rows:
        raise ValueError("series must hold at least one entry")
    rows = [row.unsqueeze(0) if row.ndim == 1 else row for row in rows]
    if any(row.ndim != 2 for row in rows):
        raise ValueError("each series must be shaped (T,) or (V, T)")

    # Allocating on the first entry's device keeps a batch of accelerator
    # tensors off the CPU round-trip that predict would otherwise have to undo.
    device = rows[0].device
    width = max(row.shape[-1] for row in rows)
    # One entry per output row: which series it came from, and how much of it is real.
    group_ids = torch.tensor([g for g, row in enumerate(rows) for _ in range(row.shape[0])], device=device)
    row_lengths = torch.tensor([row.shape[-1] for row in rows for _ in range(row.shape[0])], device=device)
    context = torch.zeros((group_ids.shape[0], width), dtype=torch.float32, device=device)
    at = 0
    for row in rows:
        context[at : at + row.shape[0], width - row.shape[-1] :] = row.to(device)
        at += row.shape[0]

    # Two layers, in this order: what the data says, then what batching added.
    # A NaN is an absent observation wherever it appears; the cells that only
    # widen a shorter entry hold no observation at all, so they overwrite it.
    mask = mask_nan_as_missing(context)
    padding = torch.arange(width, device=device) < (width - row_lengths).unsqueeze(1)
    return context, mask.masked_fill_(padding, MaskType.PAD), group_ids
