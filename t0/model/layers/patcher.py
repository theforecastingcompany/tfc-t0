"""Layer to reshape inputs in patches"""

import torch
import torch.nn as nn
from jaxtyping import Shaped
from torch import Tensor

from t0.data import MaskType, TimeSeries, VariateType, round_up


class Patcher(nn.Module):
    """Patchify time series into contiguous, non-overlapping patches.

    ``pad`` aligns a ``TimeSeries`` to the patch grid. ``patch`` reshapes an aligned
    per-time-step tensor to per-patch form. Stateless, with no parameters and no
    state-dict keys.
    """

    def __init__(self, patch_size: int):
        super().__init__()
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        self.patch_size = patch_size

    def context_end(self, model_input: TimeSeries) -> int:
        """Index where the forecast region begins.

        The first ``WITHHELD`` timestep across target rows, or ``seq_len`` when nothing
        is withheld.
        """
        is_target = model_input.variate_type[:, -1] == VariateType.TARGET
        withheld = (model_input.mask[is_target] == MaskType.WITHHELD).any(dim=0)
        # One host sync. argmax is only meaningful when something is withheld,
        # otherwise the whole series is context.
        seq_len = torch.tensor(model_input.seq_len, device=withheld.device)
        return int(torch.where(withheld.any(), withheld.to(torch.uint8).argmax(), seq_len))

    def pad(self, model_input: TimeSeries) -> TimeSeries:
        """Pad a series so the context and the forecast region both occupy whole patches.

        The model predicts one whole patch at a time, so both ends have to line up. Padding
        on the left extends the context backwards until it starts on a patch boundary, which
        holds the patch grid fixed under the forecast window. Padding on the right fills out
        the forecast region, because reading ``H`` future steps means reading
        ``ceil(H / patch_size)`` whole patches after the context.

        Left padding is inert: ``MaskType.PAD`` (1), value 0.0 and ``-1`` sentinels. Right
        padding belongs to the forecast region, so it carries each row's own group id and
        variate type, and is ``MaskType.WITHHELD`` (4) on target rows and ``PAD`` elsewhere.

        Reading the target row's mask, where 0 is an observation, 1 is padding and 4 is a
        step to predict -- a 3-step context with nothing to predict pads on the left only::

            >>> import torch
            >>> from t0.data import TimeSeries
            >>> patcher = Patcher(patch_size=2)
            >>> patcher.pad(TimeSeries.from_array(torch.zeros(1, 1, 3))).mask[0].tolist()
            [1, 0, 0, 0]

        A 4-step context is already aligned, so a 1-step horizon pads on the right only,
        rounding the forecast region up to a whole patch::

            >>> series = TimeSeries.from_array(torch.zeros(1, 1, 4), horizon=1)
            >>> patcher.pad(series).mask[0].tolist()
            [0, 0, 0, 0, 4, 4]

        A 3-step context with a 1-step horizon needs both::

            >>> series = TimeSeries.from_array(torch.zeros(1, 1, 3), horizon=1)
            >>> patcher.pad(series).mask[0].tolist()
            [1, 0, 0, 0, 4, 4]

        An aligned input is returned unchanged.
        """
        context_end = self.context_end(model_input)
        horizon = model_input.seq_len - context_end
        pad_left = (-context_end) % self.patch_size
        pad_right = round_up(horizon, self.patch_size) - horizon
        if pad_left == 0 and pad_right == 0:
            return model_input

        v = model_input.variates.shape[0]
        device = model_input.device
        is_target = model_input.variate_type[:, -1] == VariateType.TARGET

        def _left(width: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            return (
                torch.zeros(v, width, device=device, dtype=model_input.variates.dtype),
                torch.full((v, width), MaskType.PAD, device=device, dtype=torch.int8),
                torch.full((v, width), -1, device=device, dtype=torch.long),
                torch.full((v, width), -1, device=device, dtype=torch.long),
            )

        def _right(width: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            mask = torch.full((v, width), MaskType.PAD, device=device, dtype=torch.int8)
            mask[is_target] = MaskType.WITHHELD
            return (
                torch.zeros(v, width, device=device, dtype=model_input.variates.dtype),
                mask,
                model_input.group_ids[:, -1:].expand(v, width),
                model_input.variate_type[:, -1:].expand(v, width),
            )

        left = _left(pad_left)
        right = _right(pad_right)
        fields = (model_input.variates, model_input.mask, model_input.group_ids, model_input.variate_type)
        variates, mask, group_ids, variate_type = (torch.cat([left[i], fields[i], right[i]], dim=-1) for i in range(4))
        return TimeSeries(variates=variates, mask=mask, group_ids=group_ids, variate_type=variate_type)

    def patch(self, x: Shaped[Tensor, "variates time"]) -> Shaped[Tensor, "variates patches patch_size"]:
        """Reshape an aligned per-time-step tensor to per-patch form."""
        return x.unflatten(-1, (-1, self.patch_size))
