"""Attention mask helpers, operating on per-patch tensors.

Naming: a ``patched_`` prefix marks a tensor reshaped into patches but still per-time-step,
``(V, P, patch_size)``; a ``patch_`` prefix marks one value per patch, ``(V, P)``. The
helpers here reduce the former to the latter, and the builder consumes only the latter.
"""

import torch
from jaxtyping import Bool, Int
from torch import Tensor

from t0.data import MaskType, VariateType


class MaskBuilder:
    """Build SDPA attention masks from per-patch ``group_ids`` and ``variate_type``.

    Two masks per forward pass:

    - **per-patch group mask** ``(P, V, V)`` — at each patch, only variates
      that share a ``group_ids`` value can attend. For inference inputs
      (one independent series per row) every row has a unique group id and
      the mask reduces to the identity.
    - **per-variate time mask** ``(V, 1, P, P)`` — causal for target /
      historical variates, bidirectional for futures.

    Convention: ``True`` = can attend, ``False`` = blocked (matches
    ``F.scaled_dot_product_attention``).
    """

    def build_group_mask(
        self, patch_group_ids: Int[Tensor, "variates patches"]
    ) -> Bool[Tensor, "patches variates variates"]:
        """Per-patch ``(P, V, V)`` mask; ``-1`` marks padding patches."""
        valid = patch_group_ids >= 0

        ids_t = patch_group_ids.T
        val_t = valid.T
        same_group = ids_t.unsqueeze(2) == ids_t.unsqueeze(1)
        both_valid = val_t.unsqueeze(2) & val_t.unsqueeze(1)
        return same_group & both_valid

    def build_time_mask(
        self,
        patch_group_ids: Int[Tensor, "variates patches"],
        patch_variate_type: Int[Tensor, "variates patches"],
        padding_mask: Bool[Tensor, "variates patches"] | None,
    ) -> Bool[Tensor, "variates 1 patches patches"]:
        """Per-variate causal time mask ``(V, 1, P, P)``.

        Causal for target / historical variates, bidirectional for futures.
        """
        seq_len = patch_group_ids.shape[1]
        device = patch_group_ids.device

        valid_patch = patch_group_ids >= 0
        same_doc = (
            (patch_group_ids.unsqueeze(2) == patch_group_ids.unsqueeze(1))
            & valid_patch.unsqueeze(2)
            & valid_patch.unsqueeze(1)
        )
        causal = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril(diagonal=0)
        is_future = patch_variate_type == VariateType.FUTURE

        mask = same_doc & causal.unsqueeze(0)
        future_query = is_future.unsqueeze(2)
        mask = torch.where(future_query, same_doc, mask)

        if padding_mask is not None:
            key_valid = ~padding_mask.unsqueeze(1)
            mask = mask & key_valid

        return mask.unsqueeze(1)

    def expand_group_mask(
        self, group_mask: Bool[Tensor, "patches variates variates"]
    ) -> Bool[Tensor, "patches 1 variates variates"]:
        """Add the head broadcast dim so the mask shape is ``(P, 1, V, V)``."""
        return group_mask.unsqueeze(1)


def compute_patch_attention_mask(
    patched_mask: Int[Tensor, "variates patches patch_size"],
) -> Bool[Tensor, "variates patches"]:
    """Boolean mask ``(V, P)`` indicating which patch to attend.

    A patch is attended if it holds at least one non-PAD time step. In the attention mask
    True means *attend*, the opposite of the ``padding_mask`` the builder takes.
    """
    all_pad = (patched_mask == MaskType.PAD).all(dim=-1)
    return ~all_pad


def reduce_patch_metadata(
    patched_metadata: Int[Tensor, "variates patches patch_size"],
    patched_mask: Int[Tensor, "variates patches patch_size"],
) -> Int[Tensor, "variates patches"]:
    """Collapse patched metadata to each patch's first non-PAD time step, or ``-1`` when fully PAD.

    Args:
        patched_metadata: Patched ``group_ids`` or ``variate_type``.
        patched_mask: Patched typed mask, the authority on which time steps are padding.

    Returns:
        One value per patch.
    """
    is_real = patched_mask != MaskType.PAD
    first_real = is_real.int().argmax(dim=-1, keepdim=True)
    reduced = patched_metadata.gather(-1, first_real).squeeze(-1)
    return reduced.masked_fill(~is_real.any(dim=-1), -1)
