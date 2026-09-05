"""Group attention elides an all-True mask instead of materializing it.

When every variate shares one group and no patch is padding, the group mask is
all-``True`` and blocks nothing. The builder returns ``None`` there so
``F.scaled_dot_product_attention`` keeps the flash kernel. These tests pin the
short-circuit and prove the elided path matches the materialized one.
"""

import torch

from t0.data import VariateType
from t0.mask import MaskBuilder
from t0.model.layers.group_attention import VariateSelfAttention


def test_build_group_mask_elides_single_group():
    """One group, no padding -> ``None`` (all-True mask is skipped)."""
    builder = MaskBuilder()
    patch_group_ids = torch.zeros(4, 3, dtype=torch.long)  # (V=4, P=3)

    assert builder.build_group_mask(patch_group_ids) is None
    assert builder.expand_group_mask(builder.build_group_mask(patch_group_ids)) is None


def test_build_group_mask_keeps_multi_group():
    """Distinct groups still materialize the ``(P, V, V)`` mask."""
    builder = MaskBuilder()
    patch_group_ids = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)  # (V=2, P=2)

    mask = builder.build_group_mask(patch_group_ids)
    assert mask is not None
    # Per patch each variate attends only itself (2x2 identity).
    expected = torch.eye(2, dtype=torch.bool).unsqueeze(0).expand(2, 2, 2)
    assert torch.equal(mask, expected)


def test_build_group_mask_keeps_padding():
    """A padding patch (``-1``) is not all-True, so the mask is materialized."""
    builder = MaskBuilder()
    patch_group_ids = torch.tensor([[0, -1], [0, 0]], dtype=torch.long)  # (V=2, P=2)

    assert builder.build_group_mask(patch_group_ids) is not None


def test_elided_and_materialized_outputs_match():
    """SDPA output is identical whether the all-True mask is passed or elided."""
    torch.manual_seed(0)
    num_variates, num_patches, embed_dim, num_heads = 6, 5, 32, 4

    attention = VariateSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0)
    attention.eval()
    x = torch.randn(num_variates, num_patches, embed_dim, dtype=torch.float32)

    # One shared group, no padding: the mask the builder now elides.
    patch_group_ids = torch.zeros(num_variates, num_patches, dtype=torch.long)
    builder = MaskBuilder()
    assert builder.build_group_mask(patch_group_ids) is None

    all_true = torch.ones(num_patches, 1, num_variates, num_variates, dtype=torch.bool)

    with torch.no_grad():
        elided = attention(x, attn_mask=None)
        materialized = attention(x, attn_mask=all_true)

    assert torch.allclose(elided, materialized, atol=1e-5)


def test_build_time_mask_unaffected():
    """The time mask keeps its causal structure; only the group mask is elided."""
    builder = MaskBuilder()
    patch_group_ids = torch.zeros(2, 3, dtype=torch.long)
    patch_variate_type = torch.full((2, 3), VariateType.TARGET, dtype=torch.long)

    time_mask = builder.build_time_mask(patch_group_ids, patch_variate_type, None)
    assert time_mask is not None
    # Target variates are causal: the first query cannot see later keys.
    assert not time_mask[0, 0, 0, 1]
