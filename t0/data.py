"""Inference-time input container and mask vocabulary."""

from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property

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


    ``WITHHELD`` marks cells the model must predict.
    """

    VALID = 0
    PAD = 1
    MISSING = 2
    CENSORED = 3
    WITHHELD = 4


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

    @classmethod
    def from_array(
        cls,
        targets: Float[Tensor, "batch time"] | Float[Tensor, "batch variates time"],
        future_covariates: Float[Tensor, "batch future_variates context_plus_horizon"] | None = None,
    ) -> "TimeSeries":
        """Build model input from a target context and optional future covariates.

        Raises:
            ValueError: if ``targets`` is not 2-/3-D, or ``future_covariates``
                is not ``(B, F, >= T)``.
        """
        if targets.ndim not in (2, 3):
            raise ValueError(f"targets must be (B, T) or (B, V, T), got shape {tuple(targets.shape)}")
        batch_size = targets.shape[0]
        n_variates = targets.shape[1] if targets.ndim == 3 else 1
        context_len = targets.shape[-1]
        device = targets.device
        sample_ids = torch.arange(batch_size, dtype=torch.long, device=device)

        # Target rows over the context window [0, T): NaN -> MISSING + zero-filled.
        n_target = batch_size * n_variates
        variates = targets.reshape(n_target, context_len)
        mask = torch.full((n_target, context_len), MaskType.VALID, dtype=torch.int8, device=device)
        missing = torch.isnan(variates)
        if missing.any():
            mask = mask.masked_fill(missing, MaskType.MISSING)
            variates = torch.nan_to_num(variates, nan=0.0)
        group_ids = sample_ids.repeat_interleave(n_variates).unsqueeze(1).expand(n_target, context_len).contiguous()
        variate_type = torch.full((n_target, context_len), VariateType.TARGET, dtype=torch.long, device=device)
        if future_covariates is None or future_covariates.shape[1] == 0:
            return cls(variates=variates, mask=mask, group_ids=group_ids, variate_type=variate_type)

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
        fut_missing = torch.isnan(fut_values)
        if fut_missing.any():
            fut_mask = fut_mask.masked_fill(fut_missing, MaskType.MISSING)
            fut_values = torch.nan_to_num(fut_values, nan=0.0)
        fut_group = sample_ids.repeat_interleave(n_future).unsqueeze(1).expand(rows, total_len).contiguous()
        fut_type = torch.full((rows, total_len), VariateType.FUTURE, dtype=torch.long, device=device)

        # Extend target rows over the horizon with WITHHELD so all rows share one width.
        horizon = total_len - context_len
        h_values = torch.zeros((n_target, horizon), dtype=variates.dtype, device=device)
        h_mask = torch.full((n_target, horizon), MaskType.WITHHELD, dtype=torch.int8, device=device)
        h_group = group_ids[:, :1].expand(n_target, horizon)
        h_type = variate_type[:, :1].expand(n_target, horizon)
        return cls(
            variates=torch.cat([torch.cat([variates, h_values], dim=1), fut_values], dim=0),
            mask=torch.cat([torch.cat([mask, h_mask], dim=1), fut_mask], dim=0),
            group_ids=torch.cat([torch.cat([group_ids, h_group], dim=1), fut_group], dim=0),
            variate_type=torch.cat([torch.cat([variate_type, h_type], dim=1), fut_type], dim=0),
        )
