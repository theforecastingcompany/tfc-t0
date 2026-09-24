import mlx.core as mx
import numpy as np

from t0_mlx.data import MaskType, VariateType
from t0_mlx.layers import PatchEncoder, QuantileHead, RMSNorm, SwiGLU, TimeAwareRotaryEmbedding, _build_attention_masks


def test_patch_encoder_shape_and_finiteness() -> None:
    encoder = PatchEncoder(embed_dim=16, patch_size=4)
    values = mx.arange(24, dtype=mx.float32).reshape(2, 3, 4)
    mask = mx.zeros((2, 3, 4), dtype=mx.int8)
    variate_type = mx.zeros((2, 3, 4), dtype=mx.int32)
    output = encoder(values, mask, variate_type)
    mx.eval(output)
    assert output.shape == (2, 3, 16)
    assert np.isfinite(np.asarray(output)).all()


def test_rms_norm_uses_checkpoint_epsilon() -> None:
    norm = RMSNorm(3)
    output = norm(mx.array([[1.0, 2.0, 3.0]], dtype=mx.float32))
    expected = np.array([[1.0, 2.0, 3.0]]) / np.sqrt(np.mean(np.square([[1.0, 2.0, 3.0]])) + 1e-8)
    np.testing.assert_allclose(np.asarray(output), expected, rtol=1e-6, atol=1e-6)


def test_swiglu_uses_gate_first_order() -> None:
    output = SwiGLU()(mx.array([[1.0, 2.0, 3.0, 4.0]], dtype=mx.float32))
    expected = np.array([[1.0 / (1.0 + np.exp(-1.0)) * 3.0, 2.0 / (1.0 + np.exp(-2.0)) * 4.0]])
    np.testing.assert_allclose(np.asarray(output), expected, rtol=1e-6, atol=1e-6)


def test_quantile_head_is_monotone() -> None:
    output = QuantileHead([0.1, 0.5, 0.9])(mx.array([[[-2.0, -1.0, 0.5]]], dtype=mx.float32))
    values = np.asarray(output)
    assert values.shape == (1, 1, 3)
    assert np.all(values[..., 1:] >= values[..., :-1])


def test_xpos_attention_scores_are_invariant_to_position_shift() -> None:
    rotary = TimeAwareRotaryEmbedding(dims=8)
    rng = np.random.default_rng(0)
    queries, keys = (mx.array(rng.normal(size=(4, 8)).astype(np.float32)) for _ in range(2))
    q, k = rotary.rotate_queries_and_keys(queries, keys)
    shifted_q, shifted_k = rotary.rotate_queries_and_keys(
        mx.pad(queries, ((16, 0), (0, 0))), mx.pad(keys, ((16, 0), (0, 0)))
    )
    np.testing.assert_allclose(np.asarray(q @ k.T), np.asarray(shifted_q[16:] @ shifted_k[16:].T), rtol=1e-5, atol=1e-5)


def test_group_mask_future_query_reads_only_future_keys() -> None:
    types = [VariateType.TARGET, VariateType.HISTORICAL, VariateType.FUTURE]
    patched_variate_type = mx.array(types, dtype=mx.int32)[:, None, None] * mx.ones((3, 2, 4), dtype=mx.int32)
    patched_group_ids = mx.zeros((3, 2, 4), dtype=mx.int32)
    patched_mask = mx.full((3, 2, 4), MaskType.VALID, dtype=mx.int8)
    _, group_mask = _build_attention_masks(patched_group_ids, patched_variate_type, patched_mask)
    expected = np.array([[True, True, True], [True, True, True], [False, False, True]])
    np.testing.assert_array_equal(np.asarray(group_mask[0, 0]), expected)


def test_left_padded_patch_keeps_the_row_type() -> None:
    encoder = PatchEncoder(embed_dim=8, patch_size=4)
    values = mx.zeros((1, 1, 4), dtype=mx.float32)
    mask = mx.array([[[MaskType.PAD, MaskType.PAD, MaskType.VALID, MaskType.VALID]]], dtype=mx.int8)
    padded_types = mx.array([[[-1, -1, VariateType.HISTORICAL, VariateType.HISTORICAL]]], dtype=mx.int32)
    historical = mx.full((1, 1, 4), VariateType.HISTORICAL, dtype=mx.int32)
    target = mx.full((1, 1, 4), VariateType.TARGET, dtype=mx.int32)
    padded = np.asarray(encoder(values, mask, padded_types))
    np.testing.assert_allclose(padded, np.asarray(encoder(values, mask, historical)))
    assert not np.allclose(padded, np.asarray(encoder(values, mask, target)))
