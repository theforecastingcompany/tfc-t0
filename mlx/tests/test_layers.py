import mlx.core as mx
import numpy as np

from t0_mlx.layers import PatchEncoder, QuantileHead, RMSNorm, SwiGLU


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
