"""Layer to reshape inputs in patches"""

import torch
import torch.nn as nn
from jaxtyping import Shaped
from torch import Tensor

from t0.data import MaskType, TimeSeries


class Patcher(nn.Module):
    """Patchify time series into contiguous, non-overlapping patches.

    ``pad`` left-pads a ``TimeSeries`` to a patch boundary (keeping the most
    recent observation at the right edge); ``patch`` reshapes any aligned
    per-cell tensor to per-patch form. Stateless: zero parameters, zero
    state-dict keys.
    """

    def __init__(self, patch_size: int):
        super().__init__()
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        self.patch_size = patch_size

    def pad(self, model_input: TimeSeries) -> TimeSeries:
        """Left-pad so ``seq_len`` is a multiple of ``patch_size``.

        Padding cells get ``MaskType.PAD`` values 0.0 and ``-1`` sentinels in
        ``group_ids`` / ``variate_type``. Returns the input unchanged when
        already aligned.
        """
        pad_len = (-model_input.seq_len) % self.patch_size
        if pad_len == 0:
            return model_input

        v = model_input.variates.shape[0]
        device = model_input.device
        pad_values = torch.zeros(v, pad_len, device=device, dtype=model_input.variates.dtype)
        pad_mask = torch.full((v, pad_len), MaskType.PAD, device=device, dtype=torch.int8)
        pad_sentinel = torch.full((v, pad_len), -1, device=device, dtype=torch.long)

        return TimeSeries(
            variates=torch.cat([pad_values, model_input.variates], dim=-1),
            mask=torch.cat([pad_mask, model_input.mask], dim=-1),
            group_ids=torch.cat([pad_sentinel, model_input.group_ids], dim=-1),
            variate_type=torch.cat([pad_sentinel, model_input.variate_type], dim=-1),
        )

    def patch(self, x: Shaped[Tensor, "variates time"]) -> Shaped[Tensor, "variates patches patch_size"]:
        """Reshape an aligned per-cell tensor to per-patch form."""
        return x.unflatten(-1, (-1, self.patch_size))
